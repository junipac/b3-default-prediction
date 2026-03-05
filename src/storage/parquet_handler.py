"""
Handler Parquet para armazenamento analítico.
  - Particionamento por empresa/ano/mês
  - Compressão snappy (leitura rápida)
  - Schema enforcement via PyArrow
  - Versioning por timestamp
  - Data lineage automático
"""

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

from config.settings import STORAGE
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ParquetHandler:
    """
    Gerencia leitura e escrita de dados no formato Parquet.
    Usa particionamento Hive-style para consultas eficientes.
    """

    _COMPRESSION = "snappy"
    _BASE_DIR = Path(STORAGE["parquet_dir"])

    def write_market_data(
        self,
        df: pd.DataFrame,
        partition_by: List[str] = None,
    ) -> Path:
        """
        Escreve dados de mercado particionados por ano/mês.
        Preserva versão anterior se conteúdo diferente.
        """
        return self._write_partitioned(
            df=df,
            dataset_name="market_data",
            partition_cols=partition_by or ["year", "month"],
        )

    def write_financial_data(
        self,
        df: pd.DataFrame,
        doc_type: str = "dfp",
    ) -> Path:
        """Escreve dados financeiros particionados por tipo e ano."""
        return self._write_partitioned(
            df=df,
            dataset_name=f"financial/{doc_type}",
            partition_cols=["source_year"],
        )

    def write_analytical_dataset(
        self,
        df: pd.DataFrame,
        dataset_name: str,
    ) -> Path:
        """Escreve dataset analítico final."""
        return self._write_versioned(df=df, dataset_name=dataset_name)

    def read_dataset(
        self,
        dataset_name: str,
        filters: Optional[List] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Lê dataset Parquet com filtros pushdown.
        Filters: formato PyArrow, ex: [('year', '=', 2023)]
        """
        path = self._BASE_DIR / dataset_name
        if not path.exists():
            logger.warning("parquet_dataset_not_found", path=str(path))
            return pd.DataFrame()

        try:
            dataset = ds.dataset(str(path), format="parquet", partitioning="hive")
            table = dataset.to_table(filter=_build_filter(filters), columns=columns)
            df = table.to_pandas()
            logger.info(
                "parquet_read",
                dataset=dataset_name,
                rows=len(df),
                columns=len(df.columns),
            )
            return df
        except Exception as e:
            logger.error("parquet_read_failed", dataset=dataset_name, error=str(e))
            return pd.DataFrame()

    def read_latest(self, dataset_name: str) -> pd.DataFrame:
        """Lê versão mais recente de um dataset versionado."""
        path = self._BASE_DIR / dataset_name
        if not path.exists():
            return pd.DataFrame()

        versions = sorted(path.glob("v=*/"), key=lambda p: p.name, reverse=True)
        if not versions:
            return self.read_dataset(dataset_name)

        latest = versions[0]
        try:
            table = pq.read_table(str(latest))
            return table.to_pandas()
        except Exception as e:
            logger.error("parquet_read_latest_failed", error=str(e))
            return pd.DataFrame()

    def _write_partitioned(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        partition_cols: List[str],
    ) -> Path:
        """Escreve com particionamento Hive e deduplicação por hash."""
        if df.empty:
            return self._BASE_DIR / dataset_name

        target_dir = self._BASE_DIR / dataset_name
        target_dir.mkdir(parents=True, exist_ok=True)

        df = df.copy()

        # Adiciona colunas de partição se não existirem
        if "year" in partition_cols and "year" not in df.columns:
            date_col = self._detect_date_column(df)
            if date_col:
                df["year"] = pd.to_datetime(df[date_col], errors="coerce").dt.year

        if "month" in partition_cols and "month" not in df.columns:
            date_col = self._detect_date_column(df)
            if date_col:
                df["month"] = pd.to_datetime(df[date_col], errors="coerce").dt.month

        # Metadados de lineage
        df["_ingested_at"] = datetime.now(timezone.utc).isoformat()

        table = pa.Table.from_pandas(df, preserve_index=False)

        pq.write_to_dataset(
            table,
            root_path=str(target_dir),
            partition_cols=partition_cols,
            compression=self._COMPRESSION,
            existing_data_behavior="overwrite_or_ignore",
        )

        self._write_lineage(target_dir, df, partition_cols)

        logger.info(
            "parquet_partitioned_write",
            dataset=dataset_name,
            rows=len(df),
            partition_cols=partition_cols,
        )
        return target_dir

    def _write_versioned(self, df: pd.DataFrame, dataset_name: str) -> Path:
        """
        Escreve versão imutável nova a cada execução.
        Mantém histórico completo de versões.
        """
        if df.empty:
            return self._BASE_DIR / dataset_name

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        content_hash = hashlib.sha256(
            pd.util.hash_pandas_object(df, index=True).values.tobytes()
        ).hexdigest()[:12]

        version_name = f"v={timestamp}__{content_hash}"
        target_dir = self._BASE_DIR / dataset_name / version_name
        target_dir.mkdir(parents=True, exist_ok=True)

        df = df.copy()
        df["_version"] = timestamp
        df["_content_hash"] = content_hash

        table = pa.Table.from_pandas(df, preserve_index=False)
        output_file = target_dir / "data.parquet"
        pq.write_table(table, str(output_file), compression=self._COMPRESSION)

        self._write_lineage(target_dir, df, [], version=timestamp)

        logger.info(
            "parquet_versioned_write",
            dataset=dataset_name,
            version=version_name,
            rows=len(df),
            file=str(output_file),
        )
        return target_dir

    def _write_lineage(
        self,
        path: Path,
        df: pd.DataFrame,
        partition_cols: List[str],
        version: Optional[str] = None,
    ) -> None:
        """Grava arquivo de lineage junto com o dataset."""
        lineage = {
            "written_at": datetime.now(timezone.utc).isoformat(),
            "rows": len(df),
            "columns": list(df.columns),
            "partition_cols": partition_cols,
            "version": version,
            "schema": {
                col: str(dtype)
                for col, dtype in df.dtypes.items()
            },
            "row_count_by_partition": (
                df.groupby(partition_cols).size().to_dict()
                if partition_cols and all(c in df.columns for c in partition_cols)
                else {}
            ),
        }

        lineage_file = path / "_lineage.json"
        lineage_file.write_text(json.dumps(lineage, indent=2), encoding="utf-8")

    @staticmethod
    def _detect_date_column(df: pd.DataFrame) -> Optional[str]:
        candidates = ["date", "reference_date", "period_end", "data", "dt_refer"]
        for col in candidates:
            if col in df.columns:
                return col
        return None


def _build_filter(filters: Optional[List]) -> Optional[ds.Expression]:
    """Converte lista de filtros simples em expressão PyArrow."""
    if not filters:
        return None
    expressions = []
    for f in filters:
        col, op, val = f
        expr = ds.field(col)
        if op == "=":
            expressions.append(expr == val)
        elif op == ">":
            expressions.append(expr > val)
        elif op == ">=":
            expressions.append(expr >= val)
        elif op == "<":
            expressions.append(expr < val)
        elif op == "<=":
            expressions.append(expr <= val)
        elif op == "in":
            expressions.append(expr.isin(val))
    if not expressions:
        return None
    result = expressions[0]
    for e in expressions[1:]:
        result = result & e
    return result
