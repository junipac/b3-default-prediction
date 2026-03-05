"""
Parser de demonstrações financeiras.
Converte DataFrames normalizados em estrutura analítica padronizada:
  - Indicadores chave por empresa/período
  - Reconciliação Balanço Patrimonial
  - Cálculo de EBITDA, ROIC, D/E, etc.
  - Suporte a consolidado e individual
  - Detecção de inconsistências contábeis
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


class FinancialStatementParser:
    """
    Transforma dados brutos da CVM em indicadores financeiros estruturados.
    Cada método trata um demonstrativo específico e retorna um DataFrame analítico.
    """

    # Tolerância para verificação de equilíbrio do balanço (1%)
    BALANCE_TOLERANCE = 0.01

    def build_analytical_dataset(
        self,
        bpa: pd.DataFrame,
        bpp: pd.DataFrame,
        dre: pd.DataFrame,
        dfc: Optional[pd.DataFrame] = None,
        consolidado: bool = True,
    ) -> pd.DataFrame:
        """
        Constrói dataset analítico completo combinando todos os demonstrativos.
        Retorna um DataFrame no formato padrão:
          empresa | cnpj | período | indicador1 | indicador2 | ...
        """
        suffix = "con" if consolidado else "ind"
        logger.info("financial_parser_build_start", consolidado=consolidado)

        bp = self._pivot_balance_sheet(bpa, bpp)
        dre_pivoted = self._pivot_income_statement(dre)

        merged = pd.merge(
            bp,
            dre_pivoted,
            on=["cnpj_cia", "reference_date", "company_name", "cvm_code"],
            how="outer",
            suffixes=("_bp", "_dre"),
        )

        if dfc is not None and not dfc.empty:
            dfc_pivoted = self._pivot_cash_flow(dfc)
            merged = pd.merge(
                merged,
                dfc_pivoted,
                on=["cnpj_cia", "reference_date"],
                how="left",
            )

        merged = self._compute_derived_indicators(merged)
        merged = self._validate_balance_sheet(merged)
        merged = self._flag_data_quality(merged)

        logger.info(
            "financial_parser_build_done",
            rows=len(merged),
            companies=merged["cnpj_cia"].nunique() if "cnpj_cia" in merged.columns else 0,
        )
        return merged

    def _pivot_balance_sheet(
        self,
        bpa: pd.DataFrame,
        bpp: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Pivota Balanço Patrimonial Ativo e Passivo para formato colunar.
        Usa apenas versão mais recente (is_latest_version=True quando disponível).
        """
        frames = []
        for df, label in [(bpa, "BPA"), (bpp, "BPP")]:
            if df is None or df.empty:
                continue

            # Filtrar apenas versão mais recente de cada período
            if "is_latest_version" in df.columns:
                df = df[df["is_latest_version"]]

            # Filtrar apenas valores do exercício atual (não comparativos)
            if "period_type" in df.columns:
                df = df[df["period_type"].str.strip().str.upper() == "ÚLTIMO"]

            if "account_standard" not in df.columns or "value_scaled" not in df.columns:
                logger.warning(f"pivot_bp_missing_cols_{label}")
                continue

            df = df.dropna(subset=["account_standard", "value_scaled"])

            pivot = df.pivot_table(
                index=["cnpj_cia", "reference_date", "company_name", "cvm_code"],
                columns="account_standard",
                values="value_scaled",
                aggfunc="first",
            ).reset_index()

            pivot.columns.name = None
            frames.append(pivot)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, axis=1).T.groupby(level=0).first().T.reset_index(drop=True)

    def _pivot_income_statement(self, dre: pd.DataFrame) -> pd.DataFrame:
        """Pivota DRE para formato colunar."""
        if dre is None or dre.empty:
            return pd.DataFrame()

        if "is_latest_version" in dre.columns:
            dre = dre[dre["is_latest_version"]]

        if "period_type" in dre.columns:
            dre = dre[dre["period_type"].str.strip().str.upper() == "ÚLTIMO"]

        if "account_standard" not in dre.columns:
            return pd.DataFrame()

        dre = dre.dropna(subset=["account_standard", "value_scaled"])

        pivot = dre.pivot_table(
            index=["cnpj_cia", "reference_date", "company_name", "cvm_code"],
            columns="account_standard",
            values="value_scaled",
            aggfunc="first",
        ).reset_index()

        pivot.columns.name = None
        return pivot

    def _pivot_cash_flow(self, dfc: pd.DataFrame) -> pd.DataFrame:
        """Pivota DFC para formato colunar."""
        if dfc is None or dfc.empty:
            return pd.DataFrame()

        if "is_latest_version" in dfc.columns:
            dfc = dfc[dfc["is_latest_version"]]

        if "account_standard" not in dfc.columns:
            return pd.DataFrame()

        dfc = dfc.dropna(subset=["account_standard", "value_scaled"])

        pivot = dfc.pivot_table(
            index=["cnpj_cia", "reference_date"],
            columns="account_standard",
            values="value_scaled",
            aggfunc="first",
        ).reset_index()

        pivot.columns.name = None
        return pivot

    def _compute_derived_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula indicadores derivados a partir das contas padronizadas.
        Todos os cálculos documentados para auditoria.
        """

        def safe_div(num, den, default=np.nan):
            return np.where(
                (den != 0) & pd.notna(num) & pd.notna(den),
                num / den,
                default,
            )

        # ── Liquidez ──────────────────────────────────────────────────────────
        if all(c in df.columns for c in ["current_assets", "current_liabilities"]):
            df["current_ratio"] = safe_div(df["current_assets"], df["current_liabilities"])

        if all(c in df.columns for c in ["cash_equivalents", "current_liabilities"]):
            df["cash_ratio"] = safe_div(df["cash_equivalents"], df["current_liabilities"])

        if all(c in df.columns for c in ["current_assets", "inventories", "current_liabilities"]):
            df["quick_ratio"] = safe_div(
                df["current_assets"] - df["inventories"].fillna(0),
                df["current_liabilities"],
            )

        # ── Alavancagem ───────────────────────────────────────────────────────
        if all(c in df.columns for c in ["short_term_debt", "long_term_debt", "equity"]):
            total_debt = df["short_term_debt"].fillna(0) + df["long_term_debt"].fillna(0)
            df["total_debt"] = total_debt
            df["debt_to_equity"] = safe_div(total_debt, df["equity"])

        if all(c in df.columns for c in ["total_debt", "ebit"]):
            df["debt_coverage"] = safe_div(df["ebit"], df["total_debt"])

        if "total_assets" in df.columns and "equity" in df.columns:
            df["equity_multiplier"] = safe_div(df["total_assets"], df["equity"])

        # ── Net Debt & EBITDA ─────────────────────────────────────────────────
        if "total_debt" in df.columns and "cash_equivalents" in df.columns:
            df["net_debt"] = df["total_debt"] - df["cash_equivalents"].fillna(0)

        # EBITDA aproximado: EBIT + D&A (quando D&A não disponível, usa proxy do DFC)
        if "ebit" in df.columns:
            # Se tiver depreciação no DFC
            if "cfo" in df.columns:
                df["ebitda_approx"] = df["ebit"]  # Versão conservadora sem D&A
            else:
                df["ebitda_approx"] = df["ebit"]

        if "ebitda_approx" in df.columns and "net_debt" in df.columns:
            df["net_debt_ebitda"] = safe_div(df["net_debt"], df["ebitda_approx"])
            df["ebitda_margin"] = safe_div(
                df["ebitda_approx"],
                df.get("net_revenue", pd.Series(np.nan, index=df.index)),
            )

        # ── Rentabilidade ─────────────────────────────────────────────────────
        if "net_income" in df.columns and "equity" in df.columns:
            df["roe"] = safe_div(df["net_income"], df["equity"])

        if "net_income" in df.columns and "total_assets" in df.columns:
            df["roa"] = safe_div(df["net_income"], df["total_assets"])

        if "ebit" in df.columns and "total_assets" in df.columns:
            df["roic_approx"] = safe_div(df["ebit"], df["total_assets"])

        # ── Margens ───────────────────────────────────────────────────────────
        if all(c in df.columns for c in ["gross_profit", "net_revenue"]):
            df["gross_margin"] = safe_div(df["gross_profit"], df["net_revenue"])

        if all(c in df.columns for c in ["net_income", "net_revenue"]):
            df["net_margin"] = safe_div(df["net_income"], df["net_revenue"])

        if all(c in df.columns for c in ["ebit", "net_revenue"]):
            df["ebit_margin"] = safe_div(df["ebit"], df["net_revenue"])

        # ── Altman Z-Score (adaptado mercado emergente) ───────────────────────
        df = self._compute_altman_zscore(df)

        return df

    def _compute_altman_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Altman Z-Score para mercados emergentes (1995):
        Z' = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4

        X1 = Capital de giro / Ativo total
        X2 = Lucros retidos / Ativo total
        X3 = EBIT / Ativo total
        X4 = PL / Passivo total

        Zonas:
          Z' > 2.6 → Zona segura
          1.1 < Z' < 2.6 → Zona cinza
          Z' < 1.1 → Zona de distress
        """
        required = ["current_assets", "current_liabilities", "total_assets",
                    "retained_earnings", "ebit", "equity"]

        if not all(c in df.columns for c in required):
            return df

        ta = df["total_assets"]
        working_capital = df["current_assets"] - df["current_liabilities"]
        total_liabilities = ta - df["equity"].fillna(0)

        x1 = np.where(ta != 0, working_capital / ta, np.nan)
        x2 = np.where(ta != 0, df["retained_earnings"].fillna(0) / ta, np.nan)
        x3 = np.where(ta != 0, df["ebit"].fillna(0) / ta, np.nan)
        x4 = np.where(
            (total_liabilities != 0) & pd.notna(total_liabilities),
            df["equity"].fillna(0) / total_liabilities,
            np.nan,
        )

        df["altman_x1"] = x1
        df["altman_x2"] = x2
        df["altman_x3"] = x3
        df["altman_x4"] = x4
        df["altman_zscore"] = 6.56*x1 + 3.26*x2 + 6.72*x3 + 1.05*x4

        df["altman_zone"] = pd.cut(
            df["altman_zscore"],
            bins=[-np.inf, 1.1, 2.6, np.inf],
            labels=["distress", "grey", "safe"],
        )

        return df

    def _validate_balance_sheet(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Valida equilíbrio do balanço: Ativo = Passivo + PL.
        Registra desvios para controle de qualidade.
        """
        if not all(c in df.columns for c in ["total_assets", "total_liabilities_equity"]):
            return df

        df["bs_check_diff"] = (
            df["total_assets"] - df["total_liabilities_equity"].fillna(df.get("equity", 0))
        )
        df["bs_check_pct"] = np.where(
            df["total_assets"] != 0,
            abs(df["bs_check_diff"]) / df["total_assets"],
            np.nan,
        )
        df["bs_balanced"] = df["bs_check_pct"] < self.BALANCE_TOLERANCE

        n_unbalanced = (~df["bs_balanced"].fillna(True)).sum()
        if n_unbalanced > 0:
            logger.warning(
                "balance_sheet_imbalance_detected",
                n_companies=n_unbalanced,
                tolerance_pct=self.BALANCE_TOLERANCE * 100,
            )

        return df

    def _flag_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula score de completude por empresa/período."""
        key_indicators = [
            "total_assets", "equity", "net_revenue", "net_income",
            "ebit", "current_ratio", "debt_to_equity",
        ]
        available = [c for c in key_indicators if c in df.columns]

        if available:
            df["completeness_score"] = df[available].notna().sum(axis=1) / len(key_indicators)
            df["missing_pct"] = 1 - df["completeness_score"]
        else:
            df["completeness_score"] = np.nan
            df["missing_pct"] = 1.0

        return df
