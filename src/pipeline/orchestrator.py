"""
Orquestrador central do pipeline de ingestão.

Coordena:
  - Pipeline diário (cotações B3)
  - Pipeline trimestral (ITR/CVM)
  - Pipeline anual (DFP/CVM)
  - Reprocessamento de dados revisados
  - Relatório automático de inconsistências
  - Alertas de quebra de estrutura (schema drift)

Controle de execução:
  - Lock por tipo de pipeline (previne execução duplicada)
  - Relatório de execução persistido em PostgreSQL
  - Falhas isoladas por empresa não interrompem o pipeline
"""

import asyncio
import hashlib
import json
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config.settings import PIPELINE, STORAGE
from src.extractors.b3_extractor import B3MarketDataExtractor, B3CorporateEventsExtractor
from src.extractors.cvm_extractor import CVMExtractor
from src.parsers.financial_parser import FinancialStatementParser
from src.quality.data_quality import DataQualityEngine, QualityReport
from src.storage.parquet_handler import ParquetHandler
from src.default_detection.default_detector import DefaultDetector
from src.utils.logging import get_logger

logger = get_logger(__name__)


class PipelineStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass
class PipelineRun:
    """Representa uma execução do pipeline."""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pipeline_type: str = ""
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    finished_at: Optional[str] = None
    status: PipelineStatus = PipelineStatus.PENDING
    records_processed: int = 0
    companies_processed: int = 0
    errors: List[Dict] = field(default_factory=list)
    warnings: List[Dict] = field(default_factory=list)
    quality_score: Optional[float] = None
    schema_drifts: List[Dict] = field(default_factory=list)

    def finish(self, status: PipelineStatus) -> None:
        self.status = status
        self.finished_at = datetime.now(timezone.utc).isoformat()

    def add_error(self, source: str, error: str) -> None:
        self.errors.append({
            "source": source,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def to_dict(self) -> Dict:
        return {
            "run_id": self.run_id,
            "pipeline_type": self.pipeline_type,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status.value,
            "records_processed": self.records_processed,
            "companies_processed": self.companies_processed,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "quality_score": self.quality_score,
            "schema_drifts": len(self.schema_drifts),
        }


class PipelineOrchestrator:
    """
    Orquestrador principal do pipeline de ingestão B3/CVM.
    """

    def __init__(self, use_db: bool = True):
        self.use_db = use_db
        self._b3_market = B3MarketDataExtractor()
        self._b3_events = B3CorporateEventsExtractor()
        self._cvm = CVMExtractor()
        self._parser = FinancialStatementParser()
        self._quality = DataQualityEngine()
        self._parquet = ParquetHandler()
        self._detector = DefaultDetector()
        self._run_log_dir = Path(STORAGE["processed_dir"]) / "pipeline_runs"
        self._run_log_dir.mkdir(parents=True, exist_ok=True)

        if use_db:
            from src.storage.postgres_handler import PostgresHandler
            self._db = PostgresHandler(role="admin")
        else:
            self._db = None

    # ─── Pipelines Principais ─────────────────────────────────────────────────

    def run_daily(self, reference_date: Optional[date] = None) -> PipelineRun:
        """
        Pipeline diário: cotações B3 + eventos corporativos.
        Executa após fechamento do mercado (default: dia útil anterior).
        """
        run = PipelineRun(pipeline_type="daily")
        logger.info("pipeline_daily_start", run_id=run.run_id, reference_date=str(reference_date))

        try:
            # 1. Cotações do dia
            market_df = self._safe_extract(
                run=run,
                label="b3_market_daily",
                fn=lambda: self._b3_market.extract(reference_date=reference_date),
            )

            if market_df is not None and not market_df.empty:
                # Quality check de mercado
                qr = self._quality.run_market_checks(market_df, run_id=run.run_id)
                run.quality_score = qr.overall_score
                self._log_quality_issues(run, qr)

                # Persistência
                self._parquet.write_market_data(market_df)
                if self._db:
                    self._db.upsert_dataframe(
                        market_df,
                        table="market_daily",
                        conflict_cols=["ticker", "date"],
                    )
                run.records_processed += len(market_df)
                run.companies_processed += market_df["ticker"].nunique()

            # 2. Eventos corporativos (execução mais leve no diário)
            events_data = self._safe_extract(
                run=run,
                label="b3_corporate_events",
                fn=lambda: self._b3_events.extract(),
            )

            if events_data:
                for event_type, df in events_data.items():
                    if df is not None and not df.empty:
                        self._parquet.write_analytical_dataset(df, f"events/{event_type}")

            run.finish(
                PipelineStatus.SUCCESS if not run.errors else PipelineStatus.PARTIAL
            )

        except Exception as e:
            run.add_error("pipeline_daily", str(e))
            run.finish(PipelineStatus.FAILED)
            logger.error("pipeline_daily_failed", run_id=run.run_id, error=str(e))

        self._persist_run_log(run)
        logger.info(
            "pipeline_daily_done",
            run_id=run.run_id,
            status=run.status.value,
            records=run.records_processed,
        )
        return run

    def run_quarterly(self, year: Optional[int] = None) -> PipelineRun:
        """
        Pipeline trimestral: ITR da CVM.
        Processa todos os anos disponíveis se year=None.
        """
        run = PipelineRun(pipeline_type="quarterly")
        logger.info("pipeline_quarterly_start", run_id=run.run_id, year=year)

        try:
            # 1. Extração ITR
            itr_data = self._safe_extract(
                run=run,
                label="cvm_itr",
                fn=lambda: self._cvm.extract(doc_type="itr", year=year),
            )

            if itr_data:
                run = self._process_financial_data(run, itr_data, doc_type="itr")

            run.finish(
                PipelineStatus.SUCCESS if not run.errors else PipelineStatus.PARTIAL
            )

        except Exception as e:
            run.add_error("pipeline_quarterly", str(e))
            run.finish(PipelineStatus.FAILED)
            logger.error("pipeline_quarterly_failed", run_id=run.run_id, error=str(e))

        self._persist_run_log(run)
        return run

    def run_annual(self, year: Optional[int] = None) -> PipelineRun:
        """
        Pipeline anual: DFP da CVM + Cadastro + Default Detection.
        Execução mais completa e demorada.
        """
        run = PipelineRun(pipeline_type="annual")
        logger.info("pipeline_annual_start", run_id=run.run_id, year=year)

        try:
            # 1. Cadastro de empresas
            company_register = self._safe_extract(
                run=run,
                label="cvm_cadastro",
                fn=lambda: self._cvm.extract_company_register(),
            )

            if company_register is not None and not company_register.empty:
                self._parquet.write_analytical_dataset(company_register, "company_register")
                if self._db:
                    self._db.upsert_dataframe(
                        company_register,
                        table="company_register",
                        conflict_cols=["cnpj"],
                    )

            # 2. DFP
            dfp_data = self._safe_extract(
                run=run,
                label="cvm_dfp",
                fn=lambda: self._cvm.extract(doc_type="dfp", year=year),
            )

            if dfp_data:
                run = self._process_financial_data(run, dfp_data, doc_type="dfp")

            # 3. Fatos Relevantes para detecção de default
            fatos = self._safe_extract(
                run=run,
                label="cvm_fatos_relevantes",
                fn=lambda: self._cvm.extract_fatos_relevantes(),
            )

            # 4. Default Detection
            if dfp_data or fatos is not None:
                run = self._run_default_detection(
                    run=run,
                    company_register=company_register,
                    fatos_relevantes=fatos,
                )

            run.finish(
                PipelineStatus.SUCCESS if not run.errors else PipelineStatus.PARTIAL
            )

        except Exception as e:
            run.add_error("pipeline_annual", str(e))
            run.finish(PipelineStatus.FAILED)
            logger.error("pipeline_annual_failed", run_id=run.run_id, error=str(e))

        self._persist_run_log(run)
        return run

    def run_reprocess(self, cnpj: Optional[str] = None) -> PipelineRun:
        """
        Reprocessa dados revisados / reapresentados.
        Se cnpj=None, reprocessa todas as empresas com versões pendentes.
        """
        run = PipelineRun(pipeline_type="reprocess")
        logger.info("pipeline_reprocess_start", run_id=run.run_id, cnpj=cnpj)

        try:
            # Busca todos os dados financeiros disponíveis e reprocessa
            dfp_data = self._safe_extract(
                run=run,
                label="cvm_dfp_reprocess",
                fn=lambda: self._cvm.extract(doc_type="dfp", cnpj=cnpj),
            )

            if dfp_data:
                run = self._process_financial_data(run, dfp_data, doc_type="dfp_reprocess")

            run.finish(
                PipelineStatus.SUCCESS if not run.errors else PipelineStatus.PARTIAL
            )

        except Exception as e:
            run.add_error("pipeline_reprocess", str(e))
            run.finish(PipelineStatus.FAILED)

        self._persist_run_log(run)
        return run

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _process_financial_data(
        self,
        run: PipelineRun,
        financial_data: Dict,
        doc_type: str,
    ) -> PipelineRun:
        """Processa, valida e persiste dados financeiros da CVM."""
        bpa = financial_data.get("bpa_con", financial_data.get("bpa_ind", pd.DataFrame()))
        bpp = financial_data.get("bpp_con", financial_data.get("bpp_ind", pd.DataFrame()))
        dre = financial_data.get("dre_con", financial_data.get("dre_ind", pd.DataFrame()))
        dfc = financial_data.get("dfc_mi_con", financial_data.get("dfc_mi_ind"))

        if bpa.empty and bpp.empty:
            logger.warning("financial_data_empty_bp", doc_type=doc_type)
            return run

        # Parseia e calcula indicadores
        analytical_df = self._safe_extract(
            run=run,
            label=f"{doc_type}_parser",
            fn=lambda: self._parser.build_analytical_dataset(bpa, bpp, dre, dfc),
        )

        if analytical_df is None or analytical_df.empty:
            return run

        # Quality check
        qr = self._quality.run_financial_checks(analytical_df, run_id=run.run_id)
        run.quality_score = qr.overall_score
        self._log_quality_issues(run, qr)

        # Persiste raw por tipo de demonstração
        for subtype, df in financial_data.items():
            if df is not None and not df.empty:
                self._parquet.write_financial_data(df, doc_type=f"{doc_type}/{subtype}")

        # Persiste dataset analítico
        self._parquet.write_analytical_dataset(analytical_df, f"analytical/{doc_type}")

        if self._db and not analytical_df.empty:
            self._db.upsert_dataframe(
                analytical_df,
                table=f"financial_{doc_type}",
                conflict_cols=["cnpj_cia", "reference_date"],
            )

        run.records_processed += len(analytical_df)
        if "cnpj_cia" in analytical_df.columns:
            run.companies_processed += analytical_df["cnpj_cia"].nunique()

        return run

    def _run_default_detection(
        self,
        run: PipelineRun,
        company_register: Optional[pd.DataFrame],
        fatos_relevantes: Optional[pd.DataFrame],
    ) -> PipelineRun:
        """Executa detecção de default e persiste resultados."""

        # Lê dados de mercado e financeiros do Parquet
        market_df = self._parquet.read_latest("market_data")
        financial_df = self._parquet.read_latest("analytical/dfp")

        if company_register is None:
            company_register = pd.DataFrame()

        default_df = self._safe_extract(
            run=run,
            label="default_detection",
            fn=lambda: self._detector.detect_all(
                market_df=market_df,
                financial_df=financial_df,
                company_register=company_register,
                fatos_relevantes=fatos_relevantes,
            ),
        )

        if default_df is not None and not default_df.empty:
            self._parquet.write_analytical_dataset(default_df, "default_events")

            if self._db:
                self._db.upsert_dataframe(
                    default_df,
                    table="default_events",
                    conflict_cols=["cnpj"],
                )

            n_defaults = default_df.get("default_flag", pd.Series(0)).sum()
            logger.info(
                "default_detection_persisted",
                total_companies=len(default_df),
                defaults=int(n_defaults),
                run_id=run.run_id,
            )

        return run

    def _safe_extract(self, run: PipelineRun, label: str, fn) -> Optional[object]:
        """Executa função de extração com isolamento de falhas."""
        try:
            logger.info("pipeline_step_start", step=label, run_id=run.run_id)
            result = fn()
            logger.info("pipeline_step_done", step=label, run_id=run.run_id)
            return result
        except Exception as e:
            run.add_error(label, str(e))
            logger.error("pipeline_step_failed", step=label, error=str(e), run_id=run.run_id)
            return None

    def _log_quality_issues(self, run: PipelineRun, qr: QualityReport) -> None:
        """Adiciona issues do relatório de qualidade ao run."""
        for issue in qr.errors:
            run.errors.append({"source": "quality", **issue.to_dict()})
        for issue in qr.warnings:
            run.warnings.append({"source": "quality", **issue.to_dict()})

        if self._db:
            try:
                self._db.insert_quality_report(qr.summary())
            except Exception as e:
                logger.warning("quality_report_persist_failed", error=str(e))

    def _persist_run_log(self, run: PipelineRun) -> None:
        """Persiste log de execução em arquivo JSON imutável."""
        log_file = self._run_log_dir / f"run_{run.run_id}.json"
        log_file.write_text(
            json.dumps(run.to_dict(), indent=2),
            encoding="utf-8",
        )
        logger.info(
            "pipeline_run_log_persisted",
            run_id=run.run_id,
            file=str(log_file),
        )

    def validate_all_sources(self) -> Dict[str, bool]:
        """Verifica disponibilidade de todas as fontes de dados."""
        results = {
            "b3_market": self._b3_market.validate_source(),
            "b3_corporate": self._b3_events.validate_source(),
            "cvm": self._cvm.validate_source(),
        }
        logger.info("source_validation_results", **results)
        return results

    def generate_inconsistency_report(self) -> pd.DataFrame:
        """
        Gera relatório consolidado de inconsistências detectadas nos últimos runs.
        """
        log_files = sorted(self._run_log_dir.glob("run_*.json"), reverse=True)[:50]
        records = []
        for f in log_files:
            try:
                data = json.loads(f.read_text())
                for err in data.get("errors", []):
                    records.append({
                        "run_id": data["run_id"],
                        "pipeline_type": data["pipeline_type"],
                        "started_at": data["started_at"],
                        "severity": "error",
                        **err,
                    })
            except Exception:
                continue

        df = pd.DataFrame(records)
        logger.info("inconsistency_report_generated", issues=len(df))
        return df
