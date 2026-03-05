"""
Sectoral Health Index (SHI).

Indicador composto de saúde setorial baseado em Z-scores padronizados de:
    - Rentabilidade (ROA, ROE)
    - Alavancagem (D/E, dívida líquida/EBITDA)
    - Liquidez (corrente, seca)
    - Volatilidade (dispersão de resultados)

Escala: [-3, +3]
    -3 = deterioração severa
     0 = neutro histórico
    +3 = fortalecimento relevante
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SHIComponents:
    """Componentes do Sectoral Health Index."""
    sector: str
    z_rentabilidade: float
    z_alavancagem: float
    z_liquidez: float
    z_volatilidade: float
    shi: float
    classification: str
    n_companies: int


class SectoralHealthIndex:
    """
    Constrói o Sectoral Health Index (SHI).

    Metodologia:
    1. Para cada setor, calcula mediana de indicadores fundamentalistas.
    2. Padroniza (Z-score) em relação à distribuição cross-sectional.
    3. Combina em índice composto ponderado.
    4. Normaliza para escala [-3, +3].
    """

    # Pesos dos componentes
    WEIGHTS = {
        'rentabilidade': 0.30,
        'alavancagem': 0.25,
        'liquidez': 0.25,
        'volatilidade': 0.20,
    }

    # Classificações
    CLASSIFICATIONS = [
        (-3.0, -2.0, "Deterioração Severa"),
        (-2.0, -1.0, "Deterioração Moderada"),
        (-1.0, -0.5, "Deterioração Leve"),
        (-0.5,  0.5, "Neutro"),
        ( 0.5,  1.0, "Melhoria Leve"),
        ( 1.0,  2.0, "Melhoria Moderada"),
        ( 2.0,  3.0, "Fortalecimento Relevante"),
    ]

    def compute(
        self,
        df: pd.DataFrame,
        historical_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Calcula SHI para todos os setores.

        Args:
            df: DataFrame analítico corrente.
            historical_df: DataFrame histórico para normalização
                          (se None, usa distribuição corrente).

        Returns:
            DataFrame com SHI e componentes por setor.
        """
        if df.empty:
            return pd.DataFrame()

        sector_col = self._find_col(df, ['sector', 'SETOR_ATIV'])
        if not sector_col:
            return pd.DataFrame()

        df = df.copy()
        df['_sector'] = df[sector_col].fillna('Não Classificado')

        # Calcular medianas setoriais dos indicadores
        sector_stats = self._compute_sector_medians(df)

        if sector_stats.empty:
            return pd.DataFrame()

        # Z-score de cada componente
        ref_df = sector_stats  # usar como referência (ou historical_df se disponível)

        # Rentabilidade: ROA e margem EBIT (maior = melhor)
        sector_stats['z_rentabilidade'] = self._zscore(
            sector_stats['roa_median'] * 0.5 + sector_stats['ebit_margin_median'] * 0.5,
            invert=False
        )

        # Alavancagem: D/E (menor = melhor → inverter)
        sector_stats['z_alavancagem'] = self._zscore(
            sector_stats['leverage_median'],
            invert=True
        )

        # Liquidez: current ratio (maior = melhor)
        sector_stats['z_liquidez'] = self._zscore(
            sector_stats['current_ratio_median'],
            invert=False
        )

        # Volatilidade: dispersão do ROA no setor (menor = melhor → inverter)
        sector_stats['z_volatilidade'] = self._zscore(
            sector_stats['roa_std'],
            invert=True
        )

        # SHI composto
        sector_stats['shi'] = (
            sector_stats['z_rentabilidade'] * self.WEIGHTS['rentabilidade']
            + sector_stats['z_alavancagem'] * self.WEIGHTS['alavancagem']
            + sector_stats['z_liquidez'] * self.WEIGHTS['liquidez']
            + sector_stats['z_volatilidade'] * self.WEIGHTS['volatilidade']
        )

        # Normalizar para [-3, +3]
        shi_std = sector_stats['shi'].std()
        if shi_std > 0:
            sector_stats['shi_normalized'] = (
                (sector_stats['shi'] - sector_stats['shi'].mean()) / shi_std
            ).clip(-3, 3)
        else:
            sector_stats['shi_normalized'] = 0.0

        # Classificação
        sector_stats['classification'] = sector_stats['shi_normalized'].apply(
            self._classify
        )

        # Cores para visualização
        sector_stats['color'] = sector_stats['shi_normalized'].apply(
            self._shi_color
        )

        return sector_stats

    def _compute_sector_medians(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula medianas dos indicadores por setor."""
        col_map = {
            'roa': self._find_col(df, ['roa', 'ROA']),
            'ebit_margin': self._find_col(df, ['ebit_margin', 'EBIT_MARGIN']),
            'leverage': self._find_col(df, ['debt_to_equity', 'DEBT_TO_EQUITY']),
            'current_ratio': self._find_col(df, ['current_ratio', 'CURRENT_RATIO']),
            'altman': self._find_col(df, ['altman_zscore', 'ALTMAN_ZSCORE']),
        }

        results = []
        for sector, group in df.groupby('_sector'):
            n = len(group)
            if n < 2:
                continue

            stats = {'sector': sector, 'n_companies': n}

            # Mediana do ROA
            if col_map['roa']:
                roa = group[col_map['roa']].dropna()
                roa = roa[roa.between(-2, 2)]
                stats['roa_median'] = roa.median() if len(roa) > 0 else 0
                stats['roa_std'] = roa.std() if len(roa) > 1 else 0
            else:
                stats['roa_median'] = 0
                stats['roa_std'] = 0

            # Margem EBIT
            if col_map['ebit_margin']:
                em = group[col_map['ebit_margin']].dropna()
                em = em[em.between(-5, 5)]
                stats['ebit_margin_median'] = em.median() if len(em) > 0 else 0
            else:
                stats['ebit_margin_median'] = 0

            # Alavancagem
            if col_map['leverage']:
                lev = group[col_map['leverage']].dropna()
                lev = lev[lev.between(-5, 30)]
                stats['leverage_median'] = lev.median() if len(lev) > 0 else 0
            else:
                stats['leverage_median'] = 0

            # Liquidez
            if col_map['current_ratio']:
                cr = group[col_map['current_ratio']].dropna()
                cr = cr[cr.between(0, 15)]
                stats['current_ratio_median'] = cr.median() if len(cr) > 0 else 0
            else:
                stats['current_ratio_median'] = 0

            # Altman Z
            if col_map['altman']:
                az = group[col_map['altman']].dropna()
                az = az[az.between(-15, 25)]
                stats['altman_median'] = az.median() if len(az) > 0 else 0
            else:
                stats['altman_median'] = 0

            results.append(stats)

        return pd.DataFrame(results)

    @staticmethod
    def _zscore(series: pd.Series, invert: bool = False) -> pd.Series:
        """Calcula Z-score, opcionalmente invertido."""
        s = series.fillna(0)
        mean = s.mean()
        std = s.std()
        if std < 1e-10:
            return pd.Series(0.0, index=series.index)
        z = (s - mean) / std
        if invert:
            z = -z
        return z.clip(-3, 3)

    @staticmethod
    def _classify(shi: float) -> str:
        """Classifica valor do SHI."""
        if shi <= -2.0:
            return "Deterioração Severa"
        elif shi <= -1.0:
            return "Deterioração Moderada"
        elif shi <= -0.5:
            return "Deterioração Leve"
        elif shi <= 0.5:
            return "Neutro"
        elif shi <= 1.0:
            return "Melhoria Leve"
        elif shi <= 2.0:
            return "Melhoria Moderada"
        else:
            return "Fortalecimento Relevante"

    @staticmethod
    def _shi_color(shi: float) -> str:
        """Retorna cor hexadecimal para o valor do SHI."""
        if shi <= -2.0:
            return "#d32f2f"  # vermelho escuro
        elif shi <= -1.0:
            return "#f44336"  # vermelho
        elif shi <= -0.5:
            return "#ff9800"  # laranja
        elif shi <= 0.5:
            return "#9e9e9e"  # cinza (neutro)
        elif shi <= 1.0:
            return "#8bc34a"  # verde claro
        elif shi <= 2.0:
            return "#4caf50"  # verde
        else:
            return "#2e7d32"  # verde escuro

    @staticmethod
    def _find_col(df: pd.DataFrame, candidates: list) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        lower_cols = {col.lower(): col for col in df.columns}
        for c in candidates:
            if c.lower() in lower_cols:
                return lower_cols[c.lower()]
        return None
