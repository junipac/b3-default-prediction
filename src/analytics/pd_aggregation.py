"""
Agregação Setorial da PD (Merton e Scoring).

Consolida PDs individuais em métricas setoriais:
    - Média ponderada da PD
    - Mediana
    - Percentil 90
    - Distribuição por rating bucket
    - Evolução temporal
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PDAggregation:
    """Agregador de PD por setor e rating."""

    RATING_BUCKETS = [
        (0.0, 0.0003, "AAA"),
        (0.0003, 0.001, "AA"),
        (0.001, 0.005, "A"),
        (0.005, 0.02, "BBB"),
        (0.02, 0.05, "BB"),
        (0.05, 0.15, "B"),
        (0.15, 0.35, "CCC"),
        (0.35, 0.60, "CC"),
        (0.60, 1.0, "D"),
    ]

    INVESTMENT_GRADE = {"AAA", "AA", "A", "BBB"}

    def aggregate_by_sector(
        self,
        df: pd.DataFrame,
        pd_col: str = 'pd_merton',
        sector_col: str = 'sector',
        weight_col: Optional[str] = 'total_assets',
    ) -> pd.DataFrame:
        """
        Agrega PD por setor.

        Returns:
            DataFrame com: sector, n, pd_media, pd_mediana, pd_p90, pd_p95,
            pd_min, pd_max, pd_std, pd_ponderada, pct_investment_grade,
            pct_distress, dd_media
        """
        if df.empty or pd_col not in df.columns:
            return pd.DataFrame()

        df = df.copy()
        if sector_col not in df.columns:
            # Tentar variantes
            for alt in ['SETOR_ATIV', 'setor']:
                if alt in df.columns:
                    sector_col = alt
                    break

        df['_pd'] = pd.to_numeric(df[pd_col], errors='coerce')
        df = df.dropna(subset=['_pd'])
        df['_pd'] = df['_pd'].clip(0, 1)

        # Rating buckets
        df['_rating'] = df['_pd'].apply(self._assign_rating)

        results = []
        for sector, group in df.groupby(sector_col):
            n = len(group)
            pd_vals = group['_pd']

            metrics = {
                'sector': sector,
                'n_empresas': n,
                'pd_media': pd_vals.mean(),
                'pd_mediana': pd_vals.median(),
                'pd_p90': pd_vals.quantile(0.9),
                'pd_p95': pd_vals.quantile(0.95),
                'pd_min': pd_vals.min(),
                'pd_max': pd_vals.max(),
                'pd_std': pd_vals.std(),
            }

            # PD ponderada por ativo total
            if weight_col and weight_col in group.columns:
                w = group[weight_col].fillna(0).clip(0)
                total_w = w.sum()
                if total_w > 0:
                    metrics['pd_ponderada'] = (pd_vals * w).sum() / total_w
                else:
                    metrics['pd_ponderada'] = pd_vals.mean()
            else:
                metrics['pd_ponderada'] = pd_vals.mean()

            # % investment grade
            ig = group['_rating'].isin(self.INVESTMENT_GRADE).sum()
            metrics['pct_investment_grade'] = ig / n if n > 0 else 0

            # % distress (CCC ou pior)
            distress_ratings = {'CCC', 'CC', 'D'}
            distress = group['_rating'].isin(distress_ratings).sum()
            metrics['pct_distress'] = distress / n if n > 0 else 0

            # Distance-to-default média
            if 'distance_to_default' in group.columns:
                dd = group['distance_to_default'].dropna()
                metrics['dd_media'] = dd.median() if len(dd) > 0 else float('nan')
            else:
                metrics['dd_media'] = float('nan')

            # Ativo total do setor
            if weight_col and weight_col in group.columns:
                metrics['ativo_total_setor'] = group[weight_col].fillna(0).sum()
            else:
                metrics['ativo_total_setor'] = 0

            results.append(metrics)

        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values('pd_media', ascending=False)
        return result_df

    def rating_distribution(
        self,
        df: pd.DataFrame,
        pd_col: str = 'pd_merton',
    ) -> pd.DataFrame:
        """Distribuição por rating bucket."""
        if df.empty or pd_col not in df.columns:
            return pd.DataFrame()

        df = df.copy()
        df['_pd'] = pd.to_numeric(df[pd_col], errors='coerce')
        df = df.dropna(subset=['_pd'])
        df['rating'] = df['_pd'].apply(self._assign_rating)

        dist = df.groupby('rating').agg(
            n=('rating', 'count'),
            pd_media=('_pd', 'mean'),
            pd_mediana=('_pd', 'median'),
        ).reset_index()

        dist['pct'] = dist['n'] / dist['n'].sum() * 100

        # Ordenar por rating
        rating_order = [b[2] for b in self.RATING_BUCKETS]
        dist['_order'] = dist['rating'].map(
            {r: i for i, r in enumerate(rating_order)}
        )
        dist = dist.sort_values('_order').drop(columns=['_order'])

        return dist

    def sector_rating_matrix(
        self,
        df: pd.DataFrame,
        pd_col: str = 'pd_merton',
        sector_col: str = 'sector',
    ) -> pd.DataFrame:
        """Matriz setor × rating (contagem)."""
        if df.empty or pd_col not in df.columns:
            return pd.DataFrame()

        df = df.copy()
        df['_pd'] = pd.to_numeric(df[pd_col], errors='coerce')
        df = df.dropna(subset=['_pd'])
        df['rating'] = df['_pd'].apply(self._assign_rating)

        if sector_col not in df.columns:
            for alt in ['SETOR_ATIV', 'setor']:
                if alt in df.columns:
                    sector_col = alt
                    break

        matrix = pd.crosstab(
            df[sector_col],
            df['rating'],
        )

        # Reordenar colunas
        rating_order = [b[2] for b in self.RATING_BUCKETS]
        ordered = [r for r in rating_order if r in matrix.columns]
        matrix = matrix[ordered]

        return matrix

    def compute_concentration_metrics(
        self,
        sector_agg: pd.DataFrame,
    ) -> Dict:
        """
        Calcula métricas de concentração de risco.

        Returns:
            Dict com HHI, top 5 setores, etc.
        """
        if sector_agg.empty:
            return {}

        # HHI baseado em ativo total
        if 'ativo_total_setor' in sector_agg.columns:
            total = sector_agg['ativo_total_setor'].sum()
            if total > 0:
                shares = sector_agg['ativo_total_setor'] / total
                hhi = (shares ** 2).sum()
            else:
                hhi = 0
        else:
            hhi = 0

        # Top 5 setores por risco
        top_risk = sector_agg.nlargest(5, 'pd_media')[
            ['sector', 'pd_media', 'n_empresas']
        ].to_dict('records')

        # Top 5 setores mais saudáveis
        top_safe = sector_agg.nsmallest(5, 'pd_media')[
            ['sector', 'pd_media', 'n_empresas']
        ].to_dict('records')

        return {
            'hhi': hhi,
            'hhi_classification': self._classify_hhi(hhi),
            'n_sectors': len(sector_agg),
            'top_risk_sectors': top_risk,
            'top_safe_sectors': top_safe,
            'pd_media_geral': sector_agg['pd_media'].mean(),
            'pd_mediana_geral': sector_agg['pd_media'].median(),
        }

    def _assign_rating(self, pd_val: float) -> str:
        """Atribui rating bucket."""
        for low, high, rating in self.RATING_BUCKETS:
            if low <= pd_val < high:
                return rating
        return "D"

    @staticmethod
    def _classify_hhi(hhi: float) -> str:
        if hhi < 0.15:
            return "Baixa concentração"
        elif hhi < 0.25:
            return "Concentração moderada"
        else:
            return "Alta concentração"
