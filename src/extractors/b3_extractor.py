"""
Extrator B3 — dados de mercado, eventos societários, proventos e cadastro.

Fontes utilizadas:
  - Arquivo histórico de cotações (BOVESPA série histórica)
  - API de instrumentos listados
  - Eventos corporativos
  - Registro de empresas (ativas e canceladas)

Garante:
  - Empresas deslistadas permanecem na base (anti-survivorship bias)
  - Histórico de tickers reaproveitados resolvido via CNPJ raiz
  - Metadados de cada arquivo raw preservados com SHA-256
"""

import io
import re
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import requests

from src.extractors.base_extractor import BaseExtractor, ExtractionResult
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Colunas do arquivo de cotações histórico da B3 (layout fixo BOVESPA)
COTACOES_DTYPE = {
    "TIPREG": (0, 2),
    "DATPRE": (2, 10),
    "CODBDI": (10, 12),
    "CODNEG": (12, 24),
    "TPMERC": (24, 27),
    "NOMRES": (27, 39),
    "ESPECI": (39, 49),
    "PRAZOT": (49, 52),
    "MODREF": (52, 56),
    "PREABE": (56, 69),
    "PREMAX": (69, 82),
    "PREMIN": (82, 95),
    "PREMED": (95, 108),
    "PREULT": (108, 121),
    "PREOFC": (121, 134),
    "PREOFV": (134, 147),
    "TOTNEG": (147, 152),
    "QUATOT": (152, 170),
    "VOLTOT": (170, 188),
    "PREOFV_MELHOFV": (188, 201),
    "TOTNEG_MELHOFV": (201, 202),
    "DATVEN": (202, 210),
    "FATCOT": (210, 217),
    "PTOEXE": (217, 230),
    "CODISI": (230, 242),
    "DISMES": (242, 245),
}

COTACOES_COLS = [
    "TIPREG", "DATPRE", "CODBDI", "CODNEG", "TPMERC", "NOMRES",
    "ESPECI", "PREABE", "PREMAX", "PREMIN", "PREMED", "PREULT",
    "TOTNEG", "QUATOT", "VOLTOT", "FATCOT", "CODISI",
]


class B3MarketDataExtractor(BaseExtractor):
    """
    Captura e processa o histórico diário de cotações da B3.

    O arquivo COTAHIST_AXXXX.ZIP (anual) ou COTAHIST_DDDMMAAAA.ZIP (diário)
    contém cotações em formato de largura fixa, conforme layout oficial B3.
    """

    SOURCE_NAME = "b3"

    # URL base dos arquivos históricos
    _BASE_URL = "https://bvmf.bmfbovespa.com.br/InstDados/SerHist"

    def extract(self, reference_date: Optional[date] = None, annual: bool = False) -> pd.DataFrame:
        """
        Extrai cotações para uma data específica (diário) ou ano (anual).

        Args:
            reference_date: data alvo; se None, usa o dia útil anterior
            annual: se True, baixa arquivo anual completo

        Returns:
            DataFrame com cotações processadas, incluindo empresas deslistadas
        """
        if reference_date is None:
            reference_date = self._last_business_day()

        if annual:
            url, filename = self._annual_url(reference_date.year)
        else:
            url, filename = self._daily_url(reference_date)

        logger.info(
            "b3_market_extract_start",
            url=url,
            reference_date=str(reference_date),
            annual=annual,
        )

        response = self._get(url)
        raw_path = self.persist_raw(
            content=response.content,
            filename=filename,
            metadata={
                "reference_date": str(reference_date),
                "annual": annual,
                "url": url,
            },
        )

        df = self._parse_cotacoes_zip(response.content, reference_date)

        drift = self.detect_schema_drift("cotacoes", list(df.columns))
        if drift:
            logger.warning("cotacoes_schema_drift", drift=drift)

        logger.info(
            "b3_market_extract_done",
            rows=len(df),
            reference_date=str(reference_date),
            tickers=df["ticker"].nunique() if "ticker" in df.columns else 0,
        )
        return df

    def extract_date_range(
        self,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Extrai cotações para um intervalo de datas, combinando arquivos diários."""
        business_days = pd.bdate_range(start=start_date, end=end_date)
        frames: List[pd.DataFrame] = []

        for bd in business_days:
            try:
                df = self.extract(reference_date=bd.date(), annual=False)
                frames.append(df)
            except Exception as e:
                logger.warning(
                    "b3_daily_extract_failed",
                    date=str(bd.date()),
                    error=str(e),
                )

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=["ticker", "date"])
        logger.info(
            "b3_range_extract_done",
            start=str(start_date),
            end=str(end_date),
            total_rows=len(combined),
        )
        return combined

    def _parse_cotacoes_zip(self, content: bytes, reference_date: date) -> pd.DataFrame:
        """Descompacta e parseia arquivo de cotações no formato fixo BOVESPA."""
        records = []

        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            txt_files = [n for n in zf.namelist() if n.upper().endswith(".TXT")]
            if not txt_files:
                raise ValueError("Nenhum arquivo .TXT encontrado no ZIP de cotações")

            with zf.open(txt_files[0]) as f:
                for line in f:
                    row = line.decode("latin-1")
                    if len(row) < 3:
                        continue
                    tipreg = row[0:2].strip()
                    if tipreg != "01":  # Apenas registros de cotação
                        continue

                    record = self._parse_cotacao_line(row)
                    if record:
                        records.append(record)

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df = self._clean_cotacoes(df)
        return df

    def _parse_cotacao_line(self, line: str) -> Optional[Dict]:
        """Parseia uma linha do arquivo de cotações com tratamento de erros."""
        try:
            codbdi = line[10:12].strip()
            # Filtrar apenas mercado à vista normal (codbdi=02)
            # Mas manter todos para análise completa
            ticker = line[12:24].strip()
            if not ticker:
                return None

            data_str = line[2:10].strip()
            data = datetime.strptime(data_str, "%Y%m%d").date()

            def parse_price(s: str, decimals: int = 2) -> float:
                raw = s.strip()
                if not raw or raw == "0" * len(raw):
                    return 0.0
                return float(raw) / (10 ** decimals)

            return {
                "ticker": ticker,
                "date": data,
                "codbdi": codbdi,
                "company_name": line[27:39].strip(),
                "spec": line[39:49].strip(),
                "open": parse_price(line[56:69], 2),
                "high": parse_price(line[69:82], 2),
                "low": parse_price(line[82:95], 2),
                "avg": parse_price(line[95:108], 2),
                "close": parse_price(line[108:121], 2),
                "trades": int(line[147:152].strip() or 0),
                "volume_shares": int(line[152:170].strip() or 0),
                "volume_financial": parse_price(line[170:188], 2),
                "isin": line[230:242].strip(),
                "factor": int(line[210:217].strip() or 1),
            }
        except (ValueError, IndexError) as e:
            logger.debug("cotacao_line_parse_error", error=str(e))
            return None

    def _clean_cotacoes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpeza e normalização das cotações."""
        # Ajuste de preço por fator de cotação
        for col in ["open", "high", "low", "avg", "close"]:
            if col in df.columns:
                df[col] = df[col] / df["factor"].replace(0, 1)

        # Remover registros com preço zero (suspensão de negociação)
        df["suspended"] = df["close"] == 0

        # Converter tipos
        df["date"] = pd.to_datetime(df["date"])
        df["volume_financial"] = pd.to_numeric(df["volume_financial"], errors="coerce")

        return df.reset_index(drop=True)

    def _annual_url(self, year: int) -> Tuple[str, str]:
        filename = f"COTAHIST_A{year}.ZIP"
        url = f"{self._BASE_URL}/{filename}"
        return url, filename

    def _daily_url(self, d: date) -> Tuple[str, str]:
        filename = f"COTAHIST_D{d.strftime('%d%m%Y')}.ZIP"
        url = f"{self._BASE_URL}/{filename}"
        return url, filename

    @staticmethod
    def _last_business_day() -> date:
        today = date.today()
        offset = 1
        while True:
            candidate = today - timedelta(days=offset)
            if candidate.weekday() < 5:  # 0=segunda, 4=sexta
                return candidate
            offset += 1

    def validate_source(self) -> bool:
        try:
            d = self._last_business_day()
            _, filename = self._daily_url(d)
            url = f"{self._BASE_URL}/{filename}"
            resp = self._get(url, max_attempts=2)
            return resp.status_code == 200
        except Exception as e:
            logger.warning("b3_source_validation_failed", error=str(e))
            return False


class B3CorporateEventsExtractor(BaseExtractor):
    """
    Captura eventos societários via API B3:
    - Proventos (dividendos, JCP, bonificações)
    - Desdobramentos e grupamentos
    - Alterações de ticker
    - Cancelamento de registro
    - Fusões e incorporações
    """

    SOURCE_NAME = "b3"

    _API_BASE = "https://sistemaswebb3-listados.b3.com.br"
    _EVENTS_ENDPOINT = "/corporateEventsProxy/CorporateEventsCall/GetListedCorporateEvents"
    _PROVENTOS_ENDPOINT = "/listedCompaniesProxy/CompanyCall/GetListedCashDividends"
    _INSTRUMENTS_ENDPOINT = "/listedCompaniesProxy/CompanyCall/GetInitialCompanies"

    def extract(self, ticker: Optional[str] = None, **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Extrai todos os tipos de eventos corporativos.

        Returns:
            Dict com chaves: 'dividends', 'splits', 'ticker_changes',
                             'cancellations', 'instruments'
        """
        results = {
            "dividends": self._extract_dividends(ticker),
            "splits": self._extract_splits(ticker),
            "instruments": self._extract_instruments(),
            "ticker_history": self._extract_ticker_history(),
        }
        return results

    def _extract_dividends(self, ticker: Optional[str] = None) -> pd.DataFrame:
        """Extrai histórico completo de proventos."""
        logger.info("b3_dividends_extract_start", ticker=ticker)
        records = []

        # A API da B3 pagina por empresa; se ticker=None, precisamos iterar todas
        companies = self._get_all_companies()

        for company in companies:
            code = company.get("issuingCompany", "")
            if ticker and code.upper() != ticker.upper():
                continue
            try:
                url = f"{self._API_BASE}{self._PROVENTOS_ENDPOINT}/{code}/True/1/20/5"
                resp = self._get(url)
                data = resp.json()

                for item in data.get("results", []):
                    records.append({
                        "ticker": code,
                        "cnpj": company.get("cnpj", ""),
                        "company_name": company.get("companyName", ""),
                        "event_type": item.get("typeOfEvent", ""),
                        "ex_date": item.get("lastDatePrior", ""),
                        "payment_date": item.get("approvedOn", ""),
                        "value_per_share": item.get("rate", 0),
                        "currency": item.get("valueCash", "BRL"),
                        "asset_issued": item.get("assetIssued", ""),
                        "remarks": item.get("remarks", ""),
                    })
            except Exception as e:
                logger.warning(
                    "b3_dividend_extract_error",
                    ticker=code,
                    error=str(e),
                )

        df = pd.DataFrame(records) if records else pd.DataFrame()
        logger.info("b3_dividends_done", rows=len(df))
        return df

    def _extract_splits(self, ticker: Optional[str] = None) -> pd.DataFrame:
        """Extrai desdobramentos e grupamentos."""
        logger.info("b3_splits_extract_start", ticker=ticker)
        url = f"{self._API_BASE}{self._EVENTS_ENDPOINT}"
        records = []

        try:
            resp = self._get(url, params={"language": "pt-br", "pageNumber": 1, "pageSize": 9999})
            data = resp.json()

            for item in data.get("results", []):
                event_type = item.get("typeOfEvent", "").lower()
                if any(k in event_type for k in ["desdobramento", "grupamento", "split"]):
                    records.append({
                        "ticker": item.get("issuingCompany", ""),
                        "company_name": item.get("companyName", ""),
                        "event_type": item.get("typeOfEvent", ""),
                        "ex_date": item.get("lastDatePrior", ""),
                        "ratio": item.get("factor", 1),
                        "description": item.get("description", ""),
                    })
        except Exception as e:
            logger.warning("b3_splits_extract_error", error=str(e))

        df = pd.DataFrame(records) if records else pd.DataFrame()
        logger.info("b3_splits_done", rows=len(df))
        return df

    def _extract_instruments(self) -> pd.DataFrame:
        """
        Extrai cadastro completo de instrumentos listados — incluindo deslistados.
        Crítico para evitar survivorship bias.
        """
        logger.info("b3_instruments_extract_start")
        url = f"{self._API_BASE}{self._INSTRUMENTS_ENDPOINT}/1/9999/S"
        records = []

        try:
            resp = self._get(url)
            raw = resp.content
            self.persist_raw(
                content=raw,
                filename="b3_instruments.json",
                metadata={"url": url},
            )
            data = resp.json()

            for item in data.get("results", []):
                records.append({
                    "ticker": item.get("issuingCompany", "").strip(),
                    "company_name": item.get("companyName", "").strip(),
                    "cnpj": self._clean_cnpj(item.get("cnpj", "")),
                    "sector": item.get("segment", ""),
                    "listing_segment": item.get("listingSegment", ""),
                    "market": item.get("market", ""),
                    "status": item.get("status", ""),
                    "code": item.get("code", ""),
                    "isin": item.get("codeCVM", ""),
                })
        except Exception as e:
            logger.error("b3_instruments_extract_error", error=str(e))

        df = pd.DataFrame(records) if records else pd.DataFrame()
        logger.info("b3_instruments_done", rows=len(df))
        return df

    def _extract_ticker_history(self) -> pd.DataFrame:
        """
        Extrai histórico de alterações de ticker para resolver ambiguidades.
        Tickers reaproveitados são um problema crítico para análise histórica.
        """
        logger.info("b3_ticker_history_extract_start")
        url = (
            f"{self._API_BASE}/listedCompaniesProxy/CompanyCall"
            f"/GetListedCodeChanges/1/9999"
        )
        records = []

        try:
            resp = self._get(url)
            data = resp.json()

            for item in data.get("results", []):
                records.append({
                    "old_ticker": item.get("oldIssuingCompany", ""),
                    "new_ticker": item.get("newIssuingCompany", ""),
                    "cnpj": self._clean_cnpj(item.get("cnpj", "")),
                    "change_date": item.get("approvedOn", ""),
                    "reason": item.get("reason", ""),
                })
        except Exception as e:
            logger.warning("b3_ticker_history_error", error=str(e))

        df = pd.DataFrame(records) if records else pd.DataFrame()
        logger.info("b3_ticker_history_done", rows=len(df))
        return df

    def _get_all_companies(self) -> List[Dict]:
        """Retorna lista de todas empresas (ativas e canceladas)."""
        url = f"{self._API_BASE}{self._INSTRUMENTS_ENDPOINT}/1/9999/S"
        try:
            resp = self._get(url)
            return resp.json().get("results", [])
        except Exception as e:
            logger.warning("b3_get_all_companies_error", error=str(e))
            return []

    @staticmethod
    def _clean_cnpj(cnpj: str) -> str:
        """Normaliza CNPJ removendo formatação e mantendo apenas dígitos."""
        return re.sub(r"\D", "", cnpj or "").zfill(14)

    def validate_source(self) -> bool:
        try:
            url = f"{self._API_BASE}{self._INSTRUMENTS_ENDPOINT}/1/1/S"
            resp = self._get(url, max_attempts=2)
            return resp.status_code == 200
        except Exception as e:
            logger.warning("b3_corporate_source_validation_failed", error=str(e))
            return False
