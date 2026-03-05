"""
Módulo de Detecção de Default e Distress Financeiro.

Detecta eventos de default via múltiplas fontes:
  1. Recuperação Judicial (Diário Oficial + fatos relevantes CVM)
  2. Cancelamento de registro (cadastro CVM)
  3. Suspensão prolongada de negociação (dados B3)
  4. Queda extrema sustentada de preço
  5. Score de distress pré-evento (Altman Z, alavancagem, liquidez)

Produz:
  - Flag binário de default (0/1)
  - Data do evento
  - Tipo do evento
  - Score de distress pré-evento (sinal antecipado para modelo de PD)
"""

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from config.settings import DEFAULT_DETECTION
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DefaultEventType(str, Enum):
    RECUPERACAO_JUDICIAL = "recuperacao_judicial"
    FALENCIA = "falencia"
    CANCELAMENTO_REGISTRO = "cancelamento_registro"
    SUSPENSAO_PROLONGADA = "suspensao_prolongada"
    QUEDA_EXTREMA = "queda_extrema"
    LIQUIDACAO_EXTRAJUDICIAL = "liquidacao_extrajudicial"
    INTERVENCAO_REGULATORIA = "intervencao_regulatoria"


@dataclass
class DefaultEvent:
    """Evento de default identificado."""
    cnpj: str
    ticker: Optional[str]
    company_name: str
    event_type: DefaultEventType
    event_date: date
    source: str
    description: str
    confidence: float  # 0.0 a 1.0
    raw_text: Optional[str] = None
    identified_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict:
        return {
            "cnpj": self.cnpj,
            "ticker": self.ticker,
            "company_name": self.company_name,
            "event_type": self.event_type.value,
            "event_date": str(self.event_date),
            "source": self.source,
            "description": self.description,
            "confidence": self.confidence,
            "identified_at": self.identified_at,
        }


class DefaultDetector:
    """
    Detecta e classifica eventos de default de empresas listadas na B3.
    Combina regras determinísticas com score de distress financeiro.
    """

    SUSPENSION_DAYS = DEFAULT_DETECTION["suspension_days_threshold"]
    PRICE_DROP_THRESHOLD = DEFAULT_DETECTION["price_drop_threshold"]
    PRICE_DROP_WINDOW = DEFAULT_DETECTION["price_drop_window_days"]
    DISTRESS_THRESHOLD = DEFAULT_DETECTION["distress_score_threshold"]
    KEYWORDS_RJ = DEFAULT_DETECTION["keywords_recuperacao_judicial"]

    def detect_all(
        self,
        market_df: pd.DataFrame,
        financial_df: pd.DataFrame,
        company_register: pd.DataFrame,
        fatos_relevantes: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Executa detecção completa de defaults combinando todas as fontes.

        Args:
            market_df: cotações históricas (ticker, date, close, volume_financial, suspended)
            financial_df: indicadores financeiros (cnpj, altman_zscore, current_ratio, ...)
            company_register: cadastro CVM (cnpj, status, cancellation_date, ...)
            fatos_relevantes: textos de fatos relevantes publicados na CVM

        Returns:
            DataFrame com uma linha por empresa, contendo:
            cnpj | ticker | default_flag | event_type | event_date | distress_score | ...
        """
        logger.info("default_detection_start")

        events: List[DefaultEvent] = []

        # 1. Cancelamentos formais de registro
        events += self._detect_cancellations(company_register)

        # 2. Suspensão prolongada de negociação
        events += self._detect_suspension(market_df, company_register)

        # 3. Queda extrema sustentada de preço
        events += self._detect_price_crash(market_df)

        # 4. Recuperação judicial / falência via fatos relevantes
        if fatos_relevantes is not None and not fatos_relevantes.empty:
            events += self._detect_from_fatos_relevantes(fatos_relevantes, company_register)

        # Consolida eventos em DataFrame
        events_df = self._consolidate_events(events, company_register)

        # 5. Score de distress pré-evento
        distress_df = self._compute_distress_score(financial_df)

        # Combina
        result = self._merge_events_and_distress(events_df, distress_df)

        logger.info(
            "default_detection_done",
            total_companies=len(result),
            defaults_detected=result["default_flag"].sum() if "default_flag" in result.columns else 0,
        )
        return result

    def _detect_cancellations(
        self, company_register: pd.DataFrame
    ) -> List[DefaultEvent]:
        """Detecta cancelamentos formais de registro na CVM."""
        events = []
        if company_register is None or company_register.empty:
            return events

        if "is_cancelled" not in company_register.columns:
            return events

        cancelled = company_register[company_register["is_cancelled"] == True]

        for _, row in cancelled.iterrows():
            reason = str(row.get("cancellation_reason", "")).upper()
            cancel_date = row.get("cancellation_date", "")

            if not cancel_date or pd.isna(cancel_date):
                continue

            # Classifica tipo baseado no motivo
            event_type = DefaultEventType.CANCELAMENTO_REGISTRO
            if any(k.upper() in reason for k in ["FALÊNCIA", "FALENCIA", "INSOLVÊN"]):
                event_type = DefaultEventType.FALENCIA
            elif any(k.upper() in reason for k in ["RECUPERA", "RJ"]):
                event_type = DefaultEventType.RECUPERACAO_JUDICIAL
            elif "LIQUIDAÇÃO" in reason or "LIQUIDACAO" in reason:
                event_type = DefaultEventType.LIQUIDACAO_EXTRAJUDICIAL

            try:
                event_date = pd.to_datetime(cancel_date).date()
            except Exception:
                event_date = date.today()

            events.append(DefaultEvent(
                cnpj=str(row.get("cnpj", "")),
                ticker=row.get("ticker", None),
                company_name=str(row.get("company_name", "")),
                event_type=event_type,
                event_date=event_date,
                source="cvm_cadastro",
                description=f"Cancelamento CVM: {reason}",
                confidence=1.0,
            ))

        logger.info("default_cancellations_detected", count=len(events))
        return events

    def _detect_suspension(
        self,
        market_df: pd.DataFrame,
        company_register: pd.DataFrame,
    ) -> List[DefaultEvent]:
        """
        Detecta suspensão prolongada de negociação (>=N dias consecutivos
        ou total no último ano sem volume).
        """
        events = []
        if market_df is None or market_df.empty:
            return events
        if "ticker" not in market_df.columns or "volume_financial" not in market_df.columns:
            return events

        market_df = market_df.sort_values(["ticker", "date"])

        for ticker, group in market_df.groupby("ticker"):
            group = group.sort_values("date")
            zero_vol_days = (group["volume_financial"] == 0).sum()
            total_days = len(group)

            if total_days < 10:
                continue

            zero_pct = zero_vol_days / total_days

            # Detecta sequência consecutiva mais longa
            max_consecutive = self._max_consecutive_zeros(group["volume_financial"])

            if max_consecutive >= self.SUSPENSION_DAYS or zero_pct > 0.7:
                cnpj = self._lookup_cnpj(ticker, company_register)
                company_name = self._lookup_name(ticker, company_register)
                last_date = group["date"].max()

                events.append(DefaultEvent(
                    cnpj=cnpj,
                    ticker=ticker,
                    company_name=company_name,
                    event_type=DefaultEventType.SUSPENSAO_PROLONGADA,
                    event_date=(
                        pd.to_datetime(last_date).date()
                        if pd.notna(last_date) else date.today()
                    ),
                    source="b3_cotacoes",
                    description=(
                        f"Suspensão: {max_consecutive} dias consecutivos sem volume "
                        f"({zero_pct*100:.1f}% do período)"
                    ),
                    confidence=min(0.5 + max_consecutive / (self.SUSPENSION_DAYS * 2), 0.9),
                ))

        logger.info("default_suspension_detected", count=len(events))
        return events

    def _detect_price_crash(self, market_df: pd.DataFrame) -> List[DefaultEvent]:
        """
        Detecta queda extrema sustentada de preço (>80% em 252 dias).
        Sinal forte de distress mas não necessariamente default formal.
        """
        events = []
        if market_df is None or market_df.empty:
            return events
        if not all(c in market_df.columns for c in ["ticker", "date", "close"]):
            return events

        market_df = market_df.copy()
        market_df["date"] = pd.to_datetime(market_df["date"])
        market_df = market_df.sort_values(["ticker", "date"])

        for ticker, group in market_df.groupby("ticker"):
            group = group[group["close"] > 0].reset_index(drop=True)
            if len(group) < self.PRICE_DROP_WINDOW // 2:
                continue

            # Janela rolante de PRICE_DROP_WINDOW dias
            recent = group.tail(self.PRICE_DROP_WINDOW)
            if len(recent) < 20:
                continue

            price_start = recent["close"].iloc[0]
            price_end = recent["close"].iloc[-1]

            if price_start <= 0:
                continue

            total_return = (price_end - price_start) / price_start

            if total_return <= self.PRICE_DROP_THRESHOLD:
                events.append(DefaultEvent(
                    cnpj="",  # Será preenchido na consolidação
                    ticker=ticker,
                    company_name=ticker,
                    event_type=DefaultEventType.QUEDA_EXTREMA,
                    event_date=pd.to_datetime(recent["date"].iloc[-1]).date(),
                    source="b3_cotacoes",
                    description=(
                        f"Queda de {total_return*100:.1f}% em "
                        f"{len(recent)} dias úteis"
                    ),
                    confidence=min(abs(total_return) / abs(self.PRICE_DROP_THRESHOLD), 0.85),
                ))

        logger.info("default_price_crash_detected", count=len(events))
        return events

    def _detect_from_fatos_relevantes(
        self,
        fatos_df: pd.DataFrame,
        company_register: pd.DataFrame,
    ) -> List[DefaultEvent]:
        """
        Detecta eventos via análise textual de fatos relevantes CVM.
        Usa matching de palavras-chave com contexto regulatório brasileiro.
        """
        events = []

        text_cols = [c for c in fatos_df.columns if any(
            kw in c.lower() for kw in ["assunto", "descricao", "texto", "subject", "description"]
        )]

        if not text_cols:
            return events

        text_col = text_cols[0]

        keyword_patterns = {
            DefaultEventType.RECUPERACAO_JUDICIAL: [
                r"recupera[çc][aã]o\s+judicial",
                r"\bRJ\b",
                r"pedido\s+de\s+recupera[çc][aã]o",
                r"assembleia\s+geral\s+de\s+credores",
            ],
            DefaultEventType.FALENCIA: [
                r"fal[eê]ncia",
                r"insolvên?cia",
                r"decretou\s+a\s+quebra",
            ],
            DefaultEventType.LIQUIDACAO_EXTRAJUDICIAL: [
                r"liquida[çc][aã]o\s+extrajudicial",
                r"interven[çc][aã]o\s+do\s+banco\s+central",
            ],
        }

        for _, row in fatos_df.iterrows():
            texto = str(row.get(text_col, "")).lower()
            if not texto:
                continue

            cnpj_col = next((c for c in fatos_df.columns if "cnpj" in c.lower()), None)
            cnpj = str(row.get(cnpj_col, "")) if cnpj_col else ""
            company_name = str(row.get("company_name", row.get("DENOM_CIA", "")))

            date_col = next(
                (c for c in fatos_df.columns
                 if any(k in c.lower() for k in ["data", "date", "dt_"])),
                None
            )
            event_date = date.today()
            if date_col:
                try:
                    event_date = pd.to_datetime(row[date_col]).date()
                except Exception:
                    pass

            for event_type, patterns in keyword_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, texto, re.IGNORECASE):
                        events.append(DefaultEvent(
                            cnpj=self._clean_cnpj(cnpj),
                            ticker=None,
                            company_name=company_name,
                            event_type=event_type,
                            event_date=event_date,
                            source="cvm_fatos_relevantes",
                            description=f"Detectado via texto: '{pattern}'",
                            confidence=0.80,
                            raw_text=texto[:500],
                        ))
                        break

        logger.info("default_fatos_relevantes_detected", count=len(events))
        return events

    def _compute_distress_score(self, financial_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula score de distress financeiro pré-evento (0-1).
        Combina múltiplos indicadores com pesos calibrados.

        Score = 0 (saudável) → 1 (extremo distress)
        Threshold de alerta: > 0.7
        """
        if financial_df is None or financial_df.empty:
            return pd.DataFrame()

        df = financial_df.copy()

        # Componentes do score (todos normalizados para [0,1])
        score_components = []

        # 1. Altman Z-Score (invertido: Z baixo → distress alto)
        if "altman_zscore" in df.columns:
            z = df["altman_zscore"]
            # Z < 1.1 → score=1, Z > 2.6 → score=0
            z_score = 1 - np.clip((z - 1.1) / (2.6 - 1.1), 0, 1)
            score_components.append(("altman", z_score, 0.30))

        # 2. Dívida Líquida / EBITDA (>5x → score=1)
        if "net_debt_ebitda" in df.columns:
            nde = df["net_debt_ebitda"].fillna(0)
            nde_score = np.clip(nde / 5.0, 0, 1)
            score_components.append(("net_debt_ebitda", nde_score, 0.25))

        # 3. Liquidez corrente (<0.5 → score=1, >2.0 → score=0)
        if "current_ratio" in df.columns:
            cr = df["current_ratio"].fillna(1.0)
            cr_score = 1 - np.clip((cr - 0.5) / (2.0 - 0.5), 0, 1)
            score_components.append(("current_ratio", cr_score, 0.20))

        # 4. PL Negativo
        if "equity" in df.columns:
            neg_eq = (df["equity"] < 0).astype(float)
            score_components.append(("negative_equity", neg_eq, 0.15))

        # 5. Margem EBIT negativa
        if "ebit_margin" in df.columns:
            em = df["ebit_margin"].fillna(0)
            em_score = (em < 0).astype(float) * np.clip(abs(em) / 0.5, 0, 1)
            score_components.append(("ebit_margin", em_score, 0.10))

        if not score_components:
            df["distress_score"] = np.nan
            return df[["cnpj_cia", "reference_date", "distress_score"]]

        # Média ponderada
        total_weight = sum(w for _, _, w in score_components)
        weighted_sum = sum(
            component * (weight / total_weight)
            for _, component, weight in score_components
        )
        df["distress_score"] = np.clip(weighted_sum, 0, 1)

        df["distress_zone"] = pd.cut(
            df["distress_score"],
            bins=[0, 0.3, 0.7, 1.01],
            labels=["safe", "watch", "alert"],
            include_lowest=True,
        )

        # Mantém apenas a observação mais recente por empresa
        if "reference_date" in df.columns:
            df = df.sort_values("reference_date", ascending=False)
            df = df.drop_duplicates(subset=["cnpj_cia"], keep="first")

        cols = ["cnpj_cia", "reference_date", "distress_score", "distress_zone",
                "altman_zscore", "current_ratio", "net_debt_ebitda", "equity"]
        available_cols = [c for c in cols if c in df.columns]

        logger.info(
            "distress_score_computed",
            companies=len(df),
            high_distress=(df["distress_score"] > self.DISTRESS_THRESHOLD).sum(),
        )
        return df[available_cols]

    def _consolidate_events(
        self,
        events: List[DefaultEvent],
        company_register: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Consolida lista de eventos em DataFrame.
        Resolve ambiguidades de CNPJ por ticker.
        Uma empresa pode ter múltiplos eventos; mantém o mais grave.
        """
        if not events:
            return pd.DataFrame(columns=[
                "cnpj", "ticker", "company_name", "default_flag",
                "event_type", "event_date", "source", "confidence",
            ])

        rows = []
        for e in events:
            # Resolve CNPJ via ticker se ausente
            if not e.cnpj and e.ticker:
                e.cnpj = self._lookup_cnpj(e.ticker, company_register)

            rows.append({
                **e.to_dict(),
                "default_flag": 1,
            })

        df = pd.DataFrame(rows)

        # Prioridade de eventos (mais grave primeiro)
        priority_order = {
            DefaultEventType.FALENCIA.value: 1,
            DefaultEventType.RECUPERACAO_JUDICIAL.value: 2,
            DefaultEventType.LIQUIDACAO_EXTRAJUDICIAL.value: 3,
            DefaultEventType.CANCELAMENTO_REGISTRO.value: 4,
            DefaultEventType.SUSPENSAO_PROLONGADA.value: 5,
            DefaultEventType.QUEDA_EXTREMA.value: 6,
        }

        df["event_priority"] = df["event_type"].map(priority_order).fillna(99)
        df = df.sort_values(["cnpj", "event_priority", "confidence"], ascending=[True, True, False])
        df = df.drop_duplicates(subset=["cnpj"], keep="first")
        df = df.drop(columns=["event_priority"])

        return df.reset_index(drop=True)

    def _merge_events_and_distress(
        self,
        events_df: pd.DataFrame,
        distress_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Combina eventos de default com score de distress."""
        if events_df.empty and distress_df.empty:
            return pd.DataFrame()

        if events_df.empty:
            distress_df["default_flag"] = 0
            return distress_df

        if distress_df.empty:
            return events_df

        merged = pd.merge(
            events_df,
            distress_df,
            left_on="cnpj",
            right_on="cnpj_cia",
            how="outer",
        )

        # Empresas sem evento de default mas com distress alto
        merged["default_flag"] = merged["default_flag"].fillna(0).astype(int)

        # Alerta de distress para não-defaults
        if "distress_score" in merged.columns:
            merged["distress_alert"] = (
                (merged["default_flag"] == 0) &
                (merged["distress_score"] > self.DISTRESS_THRESHOLD)
            ).astype(int)

        return merged.reset_index(drop=True)

    @staticmethod
    def _max_consecutive_zeros(series: pd.Series) -> int:
        """Conta a maior sequência consecutiva de zeros."""
        max_streak = 0
        current = 0
        for val in series:
            if val == 0:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak

    @staticmethod
    def _lookup_cnpj(ticker: str, register: pd.DataFrame) -> str:
        if register is None or register.empty:
            return ""
        cols = [c for c in ["ticker", "issuingCompany", "code"] if c in register.columns]
        cnpj_col = next((c for c in ["cnpj", "cnpj_cia"] if c in register.columns), None)
        if not cols or not cnpj_col:
            return ""
        for col in cols:
            match = register[register[col].str.strip().str.upper() == ticker.upper()]
            if not match.empty:
                return str(match.iloc[0][cnpj_col])
        return ""

    @staticmethod
    def _lookup_name(ticker: str, register: pd.DataFrame) -> str:
        if register is None or register.empty:
            return ticker
        cols = [c for c in ["ticker", "issuingCompany", "code"] if c in register.columns]
        name_col = next(
            (c for c in ["company_name", "companyName", "DENOM_SOCIAL"]
             if c in register.columns), None
        )
        if not cols or not name_col:
            return ticker
        for col in cols:
            match = register[register[col].str.strip().str.upper() == ticker.upper()]
            if not match.empty:
                return str(match.iloc[0][name_col])
        return ticker

    @staticmethod
    def _clean_cnpj(cnpj: str) -> str:
        return re.sub(r"\D", "", str(cnpj or "")).zfill(14)
