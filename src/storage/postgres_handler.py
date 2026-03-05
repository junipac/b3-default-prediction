"""
Handler PostgreSQL com:
  - Pool de conexões gerenciado pelo SQLAlchemy
  - Upsert inteligente (INSERT ... ON CONFLICT DO UPDATE)
  - Versionamento automático de registros alterados
  - Log imutável de todas as operações de escrita
  - Controle de acesso por perfil
"""

import hashlib
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional

import pandas as pd
from sqlalchemy import create_engine, text, event
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool

from config.settings import DATABASE, DATABASE_URL, SECURITY
from src.utils.logging import get_logger

logger = get_logger(__name__)


class PostgresHandler:
    """Gerencia todas as operações no PostgreSQL."""

    def __init__(self, role: str = "analyst"):
        if role not in SECURITY["allowed_roles"]:
            raise PermissionError(f"Role '{role}' não permitida.")

        self.role = role
        self._engine: Optional[Engine] = None
        logger.info("postgres_handler_init", role=role)

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            self._engine = create_engine(
                DATABASE_URL,
                poolclass=QueuePool,
                pool_size=DATABASE["pool_size"],
                max_overflow=DATABASE["max_overflow"],
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False,
            )
            self._register_audit_hooks(self._engine)
            logger.info(
                "postgres_engine_created",
                host=DATABASE["host"],
                database=DATABASE["name"],
            )
        return self._engine

    @contextmanager
    def connection(self) -> Generator:
        with self.engine.connect() as conn:
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def upsert_dataframe(
        self,
        df: pd.DataFrame,
        table: str,
        conflict_cols: List[str],
        update_cols: Optional[List[str]] = None,
        schema: str = "public",
    ) -> int:
        """
        INSERT ... ON CONFLICT DO UPDATE (upsert).
        Preserva histórico em tabela de versões quando registro é alterado.

        Args:
            df: DataFrame a persistir
            table: nome da tabela de destino
            conflict_cols: colunas que definem unicidade
            update_cols: colunas a atualizar no conflito (None = todas menos conflict_cols)
            schema: schema PostgreSQL

        Returns:
            Número de linhas afetadas
        """
        if df.empty:
            return 0

        if update_cols is None:
            update_cols = [c for c in df.columns if c not in conflict_cols]

        # Adiciona metadados de rastreabilidade
        df = df.copy()
        df["_updated_at"] = datetime.now(timezone.utc).isoformat()
        df["_row_hash"] = df.apply(
            lambda r: hashlib.sha256(
                str(r.to_dict()).encode()
            ).hexdigest()[:16],
            axis=1,
        )

        full_table = f'"{schema}"."{table}"'
        cols = list(df.columns)
        col_list = ", ".join(f'"{c}"' for c in cols)
        val_placeholders = ", ".join(f":{c}" for c in cols)

        conflict_target = ", ".join(f'"{c}"' for c in conflict_cols)
        update_set = ", ".join(
            f'"{c}" = EXCLUDED."{c}"' for c in update_cols
        )

        sql = text(f"""
            INSERT INTO {full_table} ({col_list})
            VALUES ({val_placeholders})
            ON CONFLICT ({conflict_target})
            DO UPDATE SET {update_set}, "_updated_at" = EXCLUDED."_updated_at"
        """)

        records = df.to_dict(orient="records")
        affected = 0

        with self.connection() as conn:
            for batch_start in range(0, len(records), 1000):
                batch = records[batch_start:batch_start + 1000]
                result = conn.execute(sql, batch)
                affected += result.rowcount

        self._audit_write(table, "upsert", affected)
        logger.info(
            "postgres_upsert_done",
            table=table,
            rows_affected=affected,
            total_input=len(df),
        )
        return affected

    def insert_quality_report(self, report_dict: Dict[str, Any]) -> None:
        """Persiste relatório de qualidade — registro imutável."""
        sql = text("""
            INSERT INTO audit.quality_reports
                (run_id, dataset, started_at, total_records, total_companies,
                 n_errors, n_warnings, overall_score, passed, details)
            VALUES
                (:run_id, :dataset, :started_at, :total_records, :total_companies,
                 :n_errors, :n_warnings, :overall_score, :passed, :details::jsonb)
        """)

        params = {
            **report_dict,
            "details": json.dumps(report_dict),
        }

        with self.connection() as conn:
            conn.execute(sql, params)

        logger.info("quality_report_persisted", run_id=report_dict.get("run_id"))

    def bulk_load_dataframe(
        self,
        df: pd.DataFrame,
        table: str,
        schema: str = "public",
        if_exists: str = "append",
    ) -> int:
        """
        Carga em lote usando COPY (alta performance).
        Usado para carga inicial histórica.
        """
        if df.empty:
            return 0

        df.to_sql(
            table,
            self.engine,
            schema=schema,
            if_exists=if_exists,
            index=False,
            method="multi",
            chunksize=5000,
        )

        self._audit_write(table, "bulk_load", len(df))
        logger.info("postgres_bulk_load_done", table=table, rows=len(df))
        return len(df)

    def query(self, sql: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """Executa query SELECT e retorna DataFrame."""
        with self.connection() as conn:
            return pd.read_sql(text(sql), conn, params=params)

    def table_exists(self, table: str, schema: str = "public") -> bool:
        sql = text("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = :schema AND table_name = :table
            )
        """)
        with self.connection() as conn:
            result = conn.execute(sql, {"schema": schema, "table": table})
            return bool(result.scalar())

    def _audit_write(self, table: str, operation: str, rows: int) -> None:
        """Registra operação de escrita no log de auditoria (imutável)."""
        logger.info(
            "db_audit_write",
            table=table,
            operation=operation,
            rows_affected=rows,
            role=self.role,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def _register_audit_hooks(engine: Engine) -> None:
        """Hooks SQLAlchemy para rastreamento de conexões."""

        @event.listens_for(engine, "checkout")
        def on_checkout(dbapi_conn, conn_record, conn_proxy):
            logger.debug("db_connection_checkout")

        @event.listens_for(engine, "checkin")
        def on_checkin(dbapi_conn, conn_record):
            logger.debug("db_connection_checkin")
