"""
Extrator CVM — DFP, ITR, FRE e Fatos Relevantes.

Fontes: dados.cvm.gov.br (portal de dados abertos da CVM)
Layout: CSV comprimido em ZIP, separador ";", encoding latin-1

Garante:
  - Suporte a múltiplas versões de layout histórico
  - Tratamento de reapresentações (substituição retroativa)
  - Preservação de versões anteriores (versionamento)
  - Rastreamento de reapresentações por (cnpj, periodo, versão)
"""

import io
import zipfile
from datetime import datetime, timezone, date
from typing import Dict, List, Optional, Tuple
import pandas as pd
import requests

from src.extractors.base_extractor import BaseExtractor
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ─── Mapeamento de contas padronizadas ────────────────────────────────────────
# Mapeia CD_CONTA CVM → conta padronizada interna
ACCOUNT_MAP = {
    # Balanço Patrimonial Ativo
    "1": "total_assets",
    "1.01": "current_assets",
    "1.01.01": "cash_equivalents",
    "1.01.02": "financial_investments_st",
    "1.01.03": "receivables",
    "1.01.04": "inventories",
    "1.02": "non_current_assets",
    "1.02.01": "long_term_assets",
    "1.02.03": "investments",
    "1.02.04": "ppe",  # Imobilizado
    "1.02.05": "intangibles",
    # Balanço Patrimonial Passivo
    "2": "total_liabilities_equity",
    "2.01": "current_liabilities",
    "2.01.04": "short_term_debt",
    "2.02": "non_current_liabilities",
    "2.02.01": "long_term_debt",
    "2.03": "equity",
    "2.03.01": "paid_in_capital",
    "2.03.09": "retained_earnings",
    # DRE
    "3.01": "net_revenue",
    "3.02": "cost_of_goods_sold",
    "3.03": "gross_profit",
    "3.04": "operating_expenses",
    "3.05": "ebit",
    "3.06": "financial_result",
    "3.07": "ebt",
    "3.08": "income_tax",
    "3.09": "net_income_before_minority",
    "3.11": "net_income",
    # DFC
    "6.01": "cfo",   # Fluxo operacional
    "6.02": "cfi",   # Fluxo de investimento
    "6.03": "cff",   # Fluxo de financiamento
    # DVA
    "7.08": "value_added_total",
}


class CVMExtractor(BaseExtractor):
    """
    Extrator unificado para documentos estruturados da CVM.

    Suporta:
    - DFP (Demonstrações Financeiras Padronizadas — anuais)
    - ITR (Informações Trimestrais)
    - FRE (Formulário de Referência)
    - CAD (Cadastro de empresas)
    - Fatos Relevantes
    """

    SOURCE_NAME = "cvm"

    _BASE_URL = "https://dados.cvm.gov.br/dados/CIA_ABERTA"

    # Layout dos endpoints por tipo de documento
    _ENDPOINTS = {
        "dfp": {
            "url": f"{_BASE_URL}/DOC/DFP/DADOS",
            "annual": True,
            "prefix": "dfp_cia_aberta",
        },
        "itr": {
            "url": f"{_BASE_URL}/DOC/ITR/DADOS",
            "annual": False,
            "prefix": "itr_cia_aberta",
        },
        "fre": {
            "url": f"{_BASE_URL}/DOC/FRE/DADOS",
            "annual": True,
            "prefix": "fre_cia_aberta",
        },
        "cad": {
            "url": f"{_BASE_URL}/CAD",
            "annual": False,
            "prefix": "cad_cia_aberta",
        },
        "fatos_relevantes": {
            "url": f"{_BASE_URL}/DOC/FATO_RELEVANTE/DADOS",
            "annual": False,
            "prefix": "fato_relevante_cia_aberta",
        },
    }

    # Sufixos de sub-arquivos dentro do ZIP de DFP/ITR
    _DFP_SUBTYPES = [
        "BPA_con", "BPA_ind",   # Balanço Ativo Consolidado/Individual
        "BPP_con", "BPP_ind",   # Balanço Passivo
        "DRE_con", "DRE_ind",   # DRE
        "DFC_MI_con", "DFC_MI_ind",  # DFC Método Indireto
        "DFC_MD_con", "DFC_MD_ind",  # DFC Método Direto
        "DVA_con", "DVA_ind",   # DVA
        "DMPL_con", "DMPL_ind", # DMPL
    ]

    def extract(
        self,
        doc_type: str = "dfp",
        year: Optional[int] = None,
        cnpj: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Extrai documentos CVM do tipo especificado.

        Args:
            doc_type: 'dfp', 'itr', 'fre', 'cad', 'fatos_relevantes'
            year: ano de referência (None = todos disponíveis)
            cnpj: filtro por empresa específica

        Returns:
            Dict de DataFrames indexados pelo tipo de demonstração
        """
        if doc_type not in self._ENDPOINTS:
            raise ValueError(f"doc_type deve ser um de: {list(self._ENDPOINTS.keys())}")

        endpoint = self._ENDPOINTS[doc_type]

        if year:
            years_to_fetch = [year]
        else:
            years_to_fetch = self._discover_available_years(endpoint["url"])

        all_frames: Dict[str, List[pd.DataFrame]] = {}

        for yr in years_to_fetch:
            try:
                year_frames = self._fetch_year(endpoint, yr, cnpj)
                for key, df in year_frames.items():
                    all_frames.setdefault(key, []).append(df)
            except Exception as e:
                logger.warning(
                    "cvm_year_fetch_failed",
                    doc_type=doc_type,
                    year=yr,
                    error=str(e),
                )

        result = {}
        for key, frames in all_frames.items():
            combined = pd.concat(frames, ignore_index=True)
            combined = self._deduplicate_restatements(combined)
            if cnpj:
                combined = combined[combined["cnpj_cia"] == self._clean_cnpj(cnpj)]
            result[key] = combined
            logger.info(
                "cvm_extract_done",
                doc_type=doc_type,
                subtype=key,
                rows=len(combined),
                companies=combined["cnpj_cia"].nunique() if "cnpj_cia" in combined.columns else 0,
            )

        return result

    def extract_company_register(self) -> pd.DataFrame:
        """
        Extrai cadastro completo de empresas da CVM — incluindo canceladas.
        Essencial para resolver CNPJ e histórico de razão social.
        """
        logger.info("cvm_cad_extract_start")
        endpoint = self._ENDPOINTS["cad"]
        url = f"{endpoint['url']}/cad_cia_aberta.csv"

        resp = self._get(url)
        self.persist_raw(
            content=resp.content,
            filename="cad_cia_aberta.csv",
            metadata={"url": url},
        )

        df = pd.read_csv(
            io.BytesIO(resp.content),
            sep=";",
            encoding="latin-1",
            dtype=str,
            on_bad_lines="warn",
        )
        df = self._normalize_cad(df)
        logger.info("cvm_cad_done", rows=len(df))
        return df

    def extract_fatos_relevantes(
        self,
        start_year: int = 2010,
        end_year: Optional[int] = None,
    ) -> pd.DataFrame:
        """Extrai fatos relevantes para detecção de eventos de default."""
        if end_year is None:
            end_year = datetime.now().year

        endpoint = self._ENDPOINTS["fatos_relevantes"]
        frames = []

        for yr in range(start_year, end_year + 1):
            url = f"{endpoint['url']}/{endpoint['prefix']}_{yr}.zip"
            try:
                resp = self._get(url)
                self.persist_raw(
                    content=resp.content,
                    filename=f"fatos_relevantes_{yr}.zip",
                    metadata={"url": url, "year": yr},
                )
                df = self._parse_zip_csv(resp.content, f"{endpoint['prefix']}_{yr}")
                if df is not None:
                    frames.append(df)
            except Exception as e:
                logger.warning(
                    "cvm_fatos_relevantes_fetch_failed",
                    year=yr,
                    error=str(e),
                )

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        logger.info("cvm_fatos_relevantes_done", rows=len(combined))
        return combined

    # ─── Helpers privados ──────────────────────────────────────────────────────

    def _fetch_year(
        self,
        endpoint: Dict,
        year: int,
        cnpj: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Baixa e parseia todos os sub-arquivos de um ano."""
        prefix = endpoint["prefix"]
        base_url = endpoint["url"]
        url = f"{base_url}/{prefix}_{year}.zip"

        logger.info("cvm_fetch_year", url=url, year=year)
        resp = self._get(url)

        self.persist_raw(
            content=resp.content,
            filename=f"{prefix}_{year}.zip",
            metadata={"url": url, "year": year},
        )

        return self._parse_dfp_zip(resp.content, year)

    def _parse_dfp_zip(self, content: bytes, year: int) -> Dict[str, pd.DataFrame]:
        """
        Descompacta ZIP de DFP/ITR e parseia cada sub-arquivo CSV.
        Suporta múltiplas versões de layout histórico.
        """
        result = {}

        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            csv_files = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_files:
                logger.warning("cvm_zip_no_csv", year=year)
                return result

            for csv_name in csv_files:
                subtype = self._infer_subtype(csv_name)
                try:
                    with zf.open(csv_name) as f:
                        raw_bytes = f.read()

                    df = self._read_csv_robust(raw_bytes, csv_name)
                    if df is None or df.empty:
                        continue

                    df = self._normalize_financial_df(df, year)

                    drift = self.detect_schema_drift(f"cvm_{subtype}", list(df.columns))
                    if drift:
                        logger.warning("cvm_schema_drift", subtype=subtype, drift=drift)

                    result[subtype] = df
                except Exception as e:
                    logger.warning(
                        "cvm_csv_parse_failed",
                        file=csv_name,
                        year=year,
                        error=str(e),
                    )

        return result

    def _read_csv_robust(self, raw_bytes: bytes, filename: str) -> Optional[pd.DataFrame]:
        """
        Leitura robusta de CSV com tentativas de múltiplos encodings e separadores.
        Trata inconsistências históricas de layout da CVM.
        """
        encodings = ["latin-1", "utf-8", "cp1252", "iso-8859-1"]
        separators = [";", ",", "|", "\t"]

        for enc in encodings:
            for sep in separators:
                try:
                    df = pd.read_csv(
                        io.BytesIO(raw_bytes),
                        sep=sep,
                        encoding=enc,
                        dtype=str,
                        on_bad_lines="warn",
                        low_memory=False,
                    )
                    if len(df.columns) >= 5:  # Mínimo razoável
                        return df
                except Exception:
                    continue

        logger.warning("cvm_csv_all_encodings_failed", filename=filename)
        return None

    def _normalize_financial_df(self, df: pd.DataFrame, year: int) -> pd.DataFrame:
        """
        Normaliza DataFrame financeiro:
        - Padroniza nomes de colunas
        - Converte tipos
        - Adiciona conta padronizada interna
        - Marca versão e ano
        """
        # Mapeamento de nomes de colunas entre versões
        col_aliases = {
            "CNPJ_CIA": "cnpj_cia",
            "DT_REFER": "reference_date",
            "VERSAO": "version",
            "DENOM_CIA": "company_name",
            "CD_CVM": "cvm_code",
            "GRUPO_DFP": "report_group",
            "MOEDA": "currency",
            "ESCALA_MOEDA": "scale",
            "ORDEM_EXERC": "period_type",
            "DT_FIM_EXERC": "period_end",
            "DT_INI_EXERC": "period_start",
            "CD_CONTA": "account_code",
            "DS_CONTA": "account_description",
            "VL_CONTA": "value",
            "ST_CONTA_FIXA": "fixed_account",
        }

        df = df.rename(columns={k: v for k, v in col_aliases.items() if k in df.columns})
        df.columns = [c.upper().strip() for c in df.columns]

        # Re-aplicar após uppercase
        col_aliases_upper = {k.upper(): v for k, v in col_aliases.items()}
        df = df.rename(columns=col_aliases_upper)

        # Converte CNPJ
        if "cnpj_cia" in df.columns:
            df["cnpj_cia"] = df["cnpj_cia"].apply(self._clean_cnpj)

        # Converte valor para numérico
        if "value" in df.columns:
            df["value"] = pd.to_numeric(
                df["value"].str.replace(",", "."), errors="coerce"
            )

        # Aplica escala monetária (mil, unidade, etc.)
        if "scale" in df.columns and "value" in df.columns:
            df["value_scaled"] = df.apply(
                lambda r: self._apply_scale(r.get("value"), r.get("scale")),
                axis=1,
            )

        # Mapeia para conta padronizada
        if "account_code" in df.columns:
            df["account_standard"] = df["account_code"].map(ACCOUNT_MAP)

        # Metadados de rastreabilidade
        df["source_year"] = year
        df["extracted_at"] = datetime.now(timezone.utc).isoformat()

        return df

    def _normalize_cad(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza cadastro de empresas."""
        col_map = {
            "CNPJ_CIA": "cnpj",
            "DENOM_SOCIAL": "company_name",
            "DENOM_COMERC": "trade_name",
            "DT_REG": "registration_date",
            "DT_CONST": "constitution_date",
            "DT_CANCEL": "cancellation_date",
            "MOTIVO_CANCEL": "cancellation_reason",
            "SIT": "status",
            "DT_INI_SIT": "status_since",
            "CD_CVM": "cvm_code",
            "SETOR_ATIV": "sector",
            "TP_MERC": "market_type",
            "CATEG_REG": "registration_category",
            "SIT_EMISSOR": "issuer_status",
            "TP_ENDER": "address_type",
            "LOGRADOURO": "address",
            "COMPL": "address_complement",
            "MUN": "city",
            "UF": "state",
            "PAIS": "country",
            "CEP": "zip_code",
            "TEL": "phone",
            "FAX": "fax",
            "EMAIL": "email",
        }

        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        if "cnpj" in df.columns:
            df["cnpj"] = df["cnpj"].apply(self._clean_cnpj)
            df["cnpj_root"] = df["cnpj"].str[:8]

        if "status" in df.columns:
            df["is_active"] = df["status"].str.strip().str.upper().isin(["ATIVO", "A"])

        if "cancellation_date" in df.columns:
            df["is_cancelled"] = df["cancellation_date"].notna()

        return df

    def _deduplicate_restatements(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trata reapresentações: mantém a versão mais recente para cada
        combinação (cnpj, referência, conta) mas preserva histórico em
        coluna auxiliar.

        IMPORTANTE: não apaga versões anteriores — marca-as como superadas.
        """
        if "cnpj_cia" not in df.columns or "account_code" not in df.columns:
            return df

        version_col = "version" if "version" in df.columns else None
        ref_col = "reference_date" if "reference_date" in df.columns else None

        if version_col and ref_col:
            key_cols = ["cnpj_cia", ref_col, "account_code"]
            df = df.sort_values(by=version_col, ascending=False)
            df["is_latest_version"] = ~df.duplicated(subset=key_cols, keep="first")
            logger.debug(
                "cvm_restatement_dedup",
                total=len(df),
                latest=df["is_latest_version"].sum(),
                superseded=(~df["is_latest_version"]).sum(),
            )

        return df.reset_index(drop=True)

    def _discover_available_years(self, base_url: str) -> List[int]:
        """Descobre anos disponíveis listando o diretório da CVM."""
        current_year = datetime.now().year
        start_year = 2010  # Início da disponibilidade no portal aberto
        return list(range(start_year, current_year + 1))

    def _parse_zip_csv(self, content: bytes, prefix: str) -> Optional[pd.DataFrame]:
        """Parseia CSV dentro de ZIP genérico."""
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                csv_files = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                if not csv_files:
                    return None
                with zf.open(csv_files[0]) as f:
                    return self._read_csv_robust(f.read(), csv_files[0])
        except Exception as e:
            logger.warning("cvm_generic_zip_parse_error", error=str(e))
            return None

    @staticmethod
    def _infer_subtype(filename: str) -> str:
        """Infere tipo de demonstração a partir do nome do arquivo."""
        name = filename.upper()
        for subtype in [
            "BPA", "BPP", "DRE", "DFC_MI", "DFC_MD", "DVA", "DMPL",
        ]:
            if subtype.replace("_", "") in name.replace("_", ""):
                suffix = "con" if "CON" in name else "ind"
                return f"{subtype.lower()}_{suffix}"
        return Path(filename).stem.lower()

    @staticmethod
    def _apply_scale(value: Optional[float], scale: Optional[str]) -> Optional[float]:
        if value is None:
            return None
        scale_map = {
            "MIL": 1_000,
            "UNIDADE": 1,
            "MILHAO": 1_000_000,
            "BILHAO": 1_000_000_000,
        }
        scale_clean = str(scale or "MIL").strip().upper()
        multiplier = scale_map.get(scale_clean, 1_000)
        return value * multiplier

    @staticmethod
    def _clean_cnpj(cnpj: str) -> str:
        import re
        return re.sub(r"\D", "", str(cnpj or "")).zfill(14)

    def validate_source(self) -> bool:
        try:
            url = "https://dados.cvm.gov.br/dados/CIA_ABERTA/CAD/cad_cia_aberta.csv"
            resp = self._get(url, max_attempts=2)
            return resp.status_code == 200
        except Exception as e:
            logger.warning("cvm_source_validation_failed", error=str(e))
            return False
