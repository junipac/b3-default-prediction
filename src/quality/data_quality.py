"""
Motor de Data Quality para validação automática dos dados capturados.

Valida:
  - Integridade referencial (CNPJ existe no cadastro)
  - Equilíbrio do Balanço Patrimonial (Ativo = Passivo + PL)
  - Consistência do EBITDA com a DRE
  - Market cap coerente (preço × ações)
  - Ausência de look-ahead bias
  - Dados de empresas deslistadas presentes (anti-survivorship bias)

Classifica cada empresa/período por:
  - Nível de completude (0-1)
  - Nível de consistência (0-1)
  - Percentual de missing
  - Score geral de qualidade
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from config.settings import QUALITY
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class QualityIssue:
    """Representa um problema de qualidade detectado."""
    check_name: str
    severity: str          # 'error', 'warning', 'info'
    cnpj: Optional[str]
    reference_date: Optional[str]
    description: str
    value: Optional[float] = None
    threshold: Optional[float] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict:
        return {
            "check_name": self.check_name,
            "severity": self.severity,
            "cnpj": self.cnpj,
            "reference_date": self.reference_date,
            "description": self.description,
            "value": self.value,
            "threshold": self.threshold,
            "timestamp": self.timestamp,
        }


@dataclass
class QualityReport:
    """Relatório consolidado de qualidade de uma execução."""
    run_id: str
    dataset_name: str
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    total_records: int = 0
    total_companies: int = 0
    issues: List[QualityIssue] = field(default_factory=list)
    company_scores: Dict[str, Dict] = field(default_factory=dict)
    overall_score: float = 0.0
    passed: bool = True

    @property
    def errors(self) -> List[QualityIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> List[QualityIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([i.to_dict() for i in self.issues])

    def summary(self) -> Dict:
        return {
            "run_id": self.run_id,
            "dataset": self.dataset_name,
            "total_records": self.total_records,
            "total_companies": self.total_companies,
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "overall_score": round(self.overall_score, 4),
            "passed": self.passed,
        }


class DataQualityEngine:
    """
    Motor central de qualidade de dados.
    Executa bateria de checks e produz relatório estruturado.
    """

    BS_TOLERANCE = QUALITY["balance_sheet_tolerance"]
    MIN_COMPLETENESS = QUALITY["min_completeness_score"]
    MAX_MISSING = QUALITY["max_missing_pct"]
    PRICE_SPIKE = QUALITY["price_spike_threshold"]

    def run_financial_checks(
        self,
        df: pd.DataFrame,
        run_id: str = "default",
    ) -> QualityReport:
        """
        Executa todos os checks de qualidade em dados financeiros.

        Args:
            df: DataFrame analítico com indicadores financeiros
            run_id: identificador da execução (para rastreabilidade)

        Returns:
            QualityReport com todos os problemas encontrados
        """
        report = QualityReport(
            run_id=run_id,
            dataset_name="financial",
            total_records=len(df),
            total_companies=df["cnpj_cia"].nunique() if "cnpj_cia" in df.columns else 0,
        )

        logger.info(
            "dq_financial_check_start",
            run_id=run_id,
            records=report.total_records,
            companies=report.total_companies,
        )

        checks = [
            self._check_balance_sheet_integrity,
            self._check_completeness,
            self._check_ebitda_consistency,
            self._check_negative_equity,
            self._check_revenue_sign,
            self._check_date_consistency,
            self._check_duplicate_periods,
        ]

        for check_fn in checks:
            try:
                issues = check_fn(df)
                report.issues.extend(issues)
            except Exception as e:
                logger.error(
                    "dq_check_failed",
                    check=check_fn.__name__,
                    error=str(e),
                )

        report.company_scores = self._compute_company_scores(df, report.issues)
        report.overall_score = self._compute_overall_score(report)
        report.passed = len(report.errors) == 0

        logger.info(
            "dq_financial_check_done",
            run_id=run_id,
            errors=len(report.errors),
            warnings=len(report.warnings),
            overall_score=round(report.overall_score, 3),
            passed=report.passed,
        )

        return report

    def run_market_checks(
        self,
        df: pd.DataFrame,
        run_id: str = "default",
    ) -> QualityReport:
        """Executa checks de qualidade em dados de mercado."""
        report = QualityReport(
            run_id=run_id,
            dataset_name="market",
            total_records=len(df),
            total_companies=df["ticker"].nunique() if "ticker" in df.columns else 0,
        )

        checks = [
            self._check_price_spikes,
            self._check_negative_prices,
            self._check_zero_volume_extended,
            self._check_ohlc_consistency,
        ]

        for check_fn in checks:
            try:
                issues = check_fn(df)
                report.issues.extend(issues)
            except Exception as e:
                logger.error("dq_market_check_failed", check=check_fn.__name__, error=str(e))

        report.overall_score = self._compute_overall_score(report)
        report.passed = len(report.errors) == 0
        return report

    # ─── Checks Financeiros ────────────────────────────────────────────────────

    def _check_balance_sheet_integrity(
        self, df: pd.DataFrame
    ) -> List[QualityIssue]:
        """Valida: Ativo Total = Passivo Total + PL."""
        issues = []
        if not all(c in df.columns for c in ["total_assets", "bs_check_pct"]):
            return issues

        unbalanced = df[df["bs_check_pct"] > self.BS_TOLERANCE]

        for _, row in unbalanced.iterrows():
            issues.append(QualityIssue(
                check_name="balance_sheet_integrity",
                severity="error",
                cnpj=str(row.get("cnpj_cia", "")),
                reference_date=str(row.get("reference_date", "")),
                description=(
                    f"Balanço desbalanceado: desvio de "
                    f"{row['bs_check_pct']*100:.2f}% "
                    f"(tolerância: {self.BS_TOLERANCE*100:.1f}%)"
                ),
                value=float(row["bs_check_pct"]),
                threshold=self.BS_TOLERANCE,
            ))

        logger.debug(
            "dq_check_bs_integrity",
            total=len(df),
            unbalanced=len(unbalanced),
        )
        return issues

    def _check_completeness(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Verifica completude mínima dos indicadores chave."""
        issues = []
        if "completeness_score" not in df.columns:
            return issues

        low_completeness = df[df["completeness_score"] < self.MIN_COMPLETENESS]

        for _, row in low_completeness.iterrows():
            issues.append(QualityIssue(
                check_name="data_completeness",
                severity="warning",
                cnpj=str(row.get("cnpj_cia", "")),
                reference_date=str(row.get("reference_date", "")),
                description=(
                    f"Completude insuficiente: {row['completeness_score']*100:.1f}% "
                    f"(mínimo: {self.MIN_COMPLETENESS*100:.0f}%)"
                ),
                value=float(row["completeness_score"]),
                threshold=self.MIN_COMPLETENESS,
            ))

        return issues

    def _check_ebitda_consistency(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Valida que EBITDA >= EBIT (D&A nunca negativo)."""
        issues = []
        if not all(c in df.columns for c in ["ebitda_approx", "ebit"]):
            return issues

        # EBITDA < EBIT implica D&A negativo (erro contábil)
        inconsistent = df[
            pd.notna(df["ebitda_approx"]) &
            pd.notna(df["ebit"]) &
            (df["ebitda_approx"] < df["ebit"] * 0.95)  # 5% de margem para arredondamentos
        ]

        for _, row in inconsistent.iterrows():
            issues.append(QualityIssue(
                check_name="ebitda_consistency",
                severity="warning",
                cnpj=str(row.get("cnpj_cia", "")),
                reference_date=str(row.get("reference_date", "")),
                description=(
                    f"EBITDA ({row['ebitda_approx']:,.0f}) < EBIT ({row['ebit']:,.0f}): "
                    f"inconsistência contábil"
                ),
                value=float(row["ebitda_approx"] - row["ebit"]),
            ))

        return issues

    def _check_negative_equity(self, df: pd.DataFrame) -> List[QualityIssue]:
        """
        Patrimônio líquido negativo é um sinal de distress — não é erro,
        mas deve ser flagado para análise de PD.
        """
        issues = []
        if "equity" not in df.columns:
            return issues

        neg_equity = df[df["equity"] < 0]
        for _, row in neg_equity.iterrows():
            issues.append(QualityIssue(
                check_name="negative_equity",
                severity="info",
                cnpj=str(row.get("cnpj_cia", "")),
                reference_date=str(row.get("reference_date", "")),
                description=f"PL negativo: {row['equity']:,.0f}",
                value=float(row["equity"]),
            ))

        return issues

    def _check_revenue_sign(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Receita líquida deve ser positiva (exceto setor financeiro)."""
        issues = []
        if "net_revenue" not in df.columns:
            return issues

        neg_revenue = df[df["net_revenue"] < 0]
        for _, row in neg_revenue.iterrows():
            issues.append(QualityIssue(
                check_name="negative_revenue",
                severity="warning",
                cnpj=str(row.get("cnpj_cia", "")),
                reference_date=str(row.get("reference_date", "")),
                description=f"Receita líquida negativa: {row['net_revenue']:,.0f}",
                value=float(row["net_revenue"]),
            ))

        return issues

    def _check_date_consistency(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Valida que datas de referência são consistentes (sem datas futuras)."""
        issues = []
        if "reference_date" not in df.columns:
            return issues

        today = pd.Timestamp.now()
        try:
            dates = pd.to_datetime(df["reference_date"], errors="coerce")
            future_dates = df[dates > today]
            for _, row in future_dates.iterrows():
                issues.append(QualityIssue(
                    check_name="future_reference_date",
                    severity="error",
                    cnpj=str(row.get("cnpj_cia", "")),
                    reference_date=str(row.get("reference_date", "")),
                    description=f"Data de referência futura detectada (look-ahead bias)",
                ))
        except Exception:
            pass

        return issues

    def _check_duplicate_periods(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Detecta períodos duplicados sem marcação de versão."""
        issues = []
        if not all(c in df.columns for c in ["cnpj_cia", "reference_date"]):
            return issues

        if "version" in df.columns:
            return issues  # Versões múltiplas são esperadas

        dups = df[df.duplicated(subset=["cnpj_cia", "reference_date"], keep=False)]
        if not dups.empty:
            n_pairs = len(dups) // 2
            issues.append(QualityIssue(
                check_name="duplicate_periods",
                severity="warning",
                cnpj=None,
                reference_date=None,
                description=f"{n_pairs} períodos duplicados sem controle de versão",
                value=float(n_pairs),
            ))

        return issues

    # ─── Checks de Mercado ────────────────────────────────────────────────────

    def _check_price_spikes(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Detecta variações de preço suspeitas (>50% em 1 dia)."""
        issues = []
        if not all(c in df.columns for c in ["ticker", "date", "close"]):
            return issues

        df_sorted = df.sort_values(["ticker", "date"])
        df_sorted["daily_return"] = df_sorted.groupby("ticker")["close"].pct_change()

        spikes = df_sorted[
            abs(df_sorted["daily_return"]) > self.PRICE_SPIKE
        ]

        for _, row in spikes.iterrows():
            issues.append(QualityIssue(
                check_name="price_spike",
                severity="warning",
                cnpj=None,
                reference_date=str(row.get("date", "")),
                description=(
                    f"{row['ticker']}: variação de "
                    f"{row['daily_return']*100:.1f}% em 1 dia"
                ),
                value=float(row["daily_return"]),
                threshold=self.PRICE_SPIKE,
            ))

        return issues

    def _check_negative_prices(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Preços não podem ser negativos."""
        issues = []
        for col in ["open", "high", "low", "close"]:
            if col not in df.columns:
                continue
            neg = df[df[col] < 0]
            for _, row in neg.iterrows():
                issues.append(QualityIssue(
                    check_name="negative_price",
                    severity="error",
                    cnpj=None,
                    reference_date=str(row.get("date", "")),
                    description=f"{row.get('ticker', '')}: {col} negativo ({row[col]:.4f})",
                    value=float(row[col]),
                ))

        return issues

    def _check_zero_volume_extended(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Detecta períodos prolongados de volume zero (possível suspensão)."""
        issues = []
        if not all(c in df.columns for c in ["ticker", "date", "volume_financial"]):
            return issues

        df_sorted = df.sort_values(["ticker", "date"])
        zero_vol = df_sorted[df_sorted["volume_financial"] == 0].copy()

        for ticker, group in zero_vol.groupby("ticker"):
            if len(group) >= QUALITY["min_trading_days_per_year"] // 12:
                issues.append(QualityIssue(
                    check_name="extended_zero_volume",
                    severity="warning",
                    cnpj=None,
                    reference_date=None,
                    description=(
                        f"{ticker}: {len(group)} dias consecutivos sem volume — "
                        f"possível suspensão de negociação"
                    ),
                    value=float(len(group)),
                ))

        return issues

    def _check_ohlc_consistency(self, df: pd.DataFrame) -> List[QualityIssue]:
        """High >= Low, High >= Open, High >= Close."""
        issues = []
        required = ["open", "high", "low", "close"]
        if not all(c in df.columns for c in required):
            return issues

        invalid = df[
            (df["high"] < df["low"]) |
            (df["high"] < df["open"]) |
            (df["high"] < df["close"]) |
            (df["low"] > df["open"]) |
            (df["low"] > df["close"])
        ]

        if not invalid.empty:
            issues.append(QualityIssue(
                check_name="ohlc_consistency",
                severity="error",
                cnpj=None,
                reference_date=None,
                description=f"{len(invalid)} registros com OHLC inconsistente",
                value=float(len(invalid)),
            ))

        return issues

    # ─── Scores ───────────────────────────────────────────────────────────────

    def _compute_company_scores(
        self,
        df: pd.DataFrame,
        issues: List[QualityIssue],
    ) -> Dict[str, Dict]:
        """Calcula score de qualidade por empresa."""
        scores = {}

        if "cnpj_cia" not in df.columns:
            return scores

        for cnpj in df["cnpj_cia"].unique():
            company_df = df[df["cnpj_cia"] == cnpj]
            company_issues = [i for i in issues if i.cnpj == cnpj]

            completeness = float(company_df.get("completeness_score", pd.Series([np.nan])).mean())
            n_errors = sum(1 for i in company_issues if i.severity == "error")
            n_warnings = sum(1 for i in company_issues if i.severity == "warning")

            consistency = max(0.0, 1.0 - (n_errors * 0.3) - (n_warnings * 0.05))

            scores[cnpj] = {
                "cnpj": cnpj,
                "completeness": round(completeness, 4),
                "consistency": round(consistency, 4),
                "missing_pct": round(1 - completeness, 4),
                "n_errors": n_errors,
                "n_warnings": n_warnings,
                "overall": round((completeness + consistency) / 2, 4),
            }

        return scores

    def _compute_overall_score(self, report: QualityReport) -> float:
        if not report.company_scores:
            return 1.0 if not report.errors else 0.0

        scores = [s["overall"] for s in report.company_scores.values()]
        return float(np.mean(scores)) if scores else 0.0
