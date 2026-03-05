"""
Módulo de Validação e Backtest do Modelo de Merton.

Implementa:
    - Backtest PD implícita vs eventos reais de default
    - Correlação PD vs recuperação judicial
    - Sensibilidade a choques macroeconômicos
    - Análise de poder discriminante (ROC/AUC, KS, Gini)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Resultado do backtest do modelo."""
    n_total: int
    n_defaults: int
    n_non_defaults: int
    default_rate: float
    avg_pd_defaults: float
    avg_pd_non_defaults: float
    pd_ratio: float            # pd_defaults / pd_non_defaults
    ks_statistic: float        # Kolmogorov-Smirnov
    gini_coefficient: float
    accuracy_ratio: float
    power_curve_area: float    # Área sob curva de poder
    decile_analysis: pd.DataFrame
    lift_top_decile: float


class ModelValidator:
    """
    Validador do modelo de PD.

    Compara PDs estimadas contra eventos reais de default
    para avaliar poder discriminante e calibração.
    """

    def backtest(
        self,
        df: pd.DataFrame,
        pd_col: str = 'pd_merton',
        default_col: str = 'default_flag',
    ) -> BacktestResult:
        """
        Executa backtest: PD estimada vs default real.

        Args:
            df: DataFrame com PD estimada e flag de default.
            pd_col: Coluna com PD estimada.
            default_col: Coluna com flag de default (0/1).

        Returns:
            BacktestResult com métricas de validação.
        """
        if df.empty or pd_col not in df.columns:
            return self._empty_result()

        valid = df.dropna(subset=[pd_col])
        valid = valid.copy()
        valid['_pd'] = pd.to_numeric(valid[pd_col], errors='coerce')

        if default_col not in valid.columns:
            # Criar proxy de default baseado em sinais de distress
            valid['_default'] = self._create_default_proxy(valid)
        else:
            valid['_default'] = valid[default_col].fillna(0).astype(int)

        valid = valid.dropna(subset=['_pd', '_default'])

        n_total = len(valid)
        n_defaults = int(valid['_default'].sum())
        n_non_defaults = n_total - n_defaults

        if n_total == 0:
            return self._empty_result()

        default_rate = n_defaults / n_total if n_total > 0 else 0

        # PD média por grupo
        defaults = valid[valid['_default'] == 1]
        non_defaults = valid[valid['_default'] == 0]

        avg_pd_def = defaults['_pd'].mean() if len(defaults) > 0 else 0
        avg_pd_nondef = non_defaults['_pd'].mean() if len(non_defaults) > 0 else 0
        pd_ratio = avg_pd_def / avg_pd_nondef if avg_pd_nondef > 0 else float('inf')

        # KS statistic
        ks = self._compute_ks(valid['_pd'], valid['_default'])

        # Gini coefficient
        gini = self._compute_gini(valid['_pd'], valid['_default'])

        # Análise por decil
        decile = self._decile_analysis(valid['_pd'], valid['_default'])

        # Lift no topo
        lift_top = 0
        if not decile.empty and default_rate > 0:
            top_decile = decile.iloc[-1]  # decil mais arriscado
            lift_top = top_decile.get('default_rate_decile', 0) / default_rate

        return BacktestResult(
            n_total=n_total,
            n_defaults=n_defaults,
            n_non_defaults=n_non_defaults,
            default_rate=default_rate,
            avg_pd_defaults=avg_pd_def,
            avg_pd_non_defaults=avg_pd_nondef,
            pd_ratio=pd_ratio,
            ks_statistic=ks,
            gini_coefficient=gini,
            accuracy_ratio=gini,  # AR ≈ Gini para modelos de crédito
            power_curve_area=0.5 + gini / 2,
            decile_analysis=decile,
            lift_top_decile=lift_top,
        )

    def sensitivity_analysis(
        self,
        df: pd.DataFrame,
        pd_col: str = 'pd_merton',
        sector_col: str = 'sector',
        shocks: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Análise de sensibilidade a choques macroeconômicos.

        Simula impacto de variações em:
        - Taxa de juros (Selic)
        - Volatilidade de mercado
        - Câmbio

        Args:
            df: DataFrame com resultados Merton.
            shocks: Dict de cenários {'nome': fator_multiplicativo}.

        Returns:
            DataFrame com PD sob diferentes cenários.
        """
        if shocks is None:
            shocks = {
                'Base': 1.0,
                'Selic +200bps': 1.15,       # PD sobe ~15%
                'Selic -200bps': 0.88,        # PD cai ~12%
                'Vol +30%': 1.25,             # PD sobe ~25%
                'Vol -30%': 0.80,             # PD cai ~20%
                'Estresse Severo': 1.50,      # PD sobe ~50%
                'Cenário Favorável': 0.70,    # PD cai ~30%
            }

        if df.empty or pd_col not in df.columns:
            return pd.DataFrame()

        results = []

        for scenario, factor in shocks.items():
            scenario_pd = df[pd_col].dropna() * factor
            scenario_pd = scenario_pd.clip(0, 1)

            results.append({
                'cenario': scenario,
                'fator': factor,
                'pd_media': scenario_pd.mean(),
                'pd_mediana': scenario_pd.median(),
                'pd_p90': scenario_pd.quantile(0.9),
                'pd_p95': scenario_pd.quantile(0.95),
                'n_distress': (scenario_pd > 0.15).sum(),
                'pct_distress': (scenario_pd > 0.15).mean(),
                'n_default': (scenario_pd > 0.50).sum(),
                'pct_default': (scenario_pd > 0.50).mean(),
            })

        return pd.DataFrame(results)

    def correlation_with_defaults(
        self,
        df: pd.DataFrame,
        pd_col: str = 'pd_merton',
        default_events: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        Correlação entre PD e eventos de default/RJ.

        Returns:
            Dict com correlações e estatísticas.
        """
        result = {
            'rank_correlation': float('nan'),
            'point_biserial': float('nan'),
            'avg_pd_rj': float('nan'),
            'avg_pd_non_rj': float('nan'),
            'n_rj': 0,
        }

        if df.empty or pd_col not in df.columns:
            return result

        # Usar proxy de distress se não houver eventos reais
        df = df.copy()
        if default_events is not None and not default_events.empty:
            # Merge eventos
            cnpj_col = None
            for c in ['cnpj', 'cnpj_cia', 'CNPJ_CIA']:
                if c in df.columns:
                    cnpj_col = c
                    break
            if cnpj_col and 'cnpj' in default_events.columns:
                defaults_set = set(default_events['cnpj'].unique())
                df['_rj'] = df[cnpj_col].isin(defaults_set).astype(int)
            else:
                df['_rj'] = self._create_default_proxy(df)
        else:
            df['_rj'] = self._create_default_proxy(df)

        rj = df[df['_rj'] == 1]
        non_rj = df[df['_rj'] == 0]

        result['n_rj'] = len(rj)
        result['avg_pd_rj'] = rj[pd_col].mean() if len(rj) > 0 else float('nan')
        result['avg_pd_non_rj'] = non_rj[pd_col].mean() if len(non_rj) > 0 else float('nan')

        # Correlação de Spearman
        pd_vals = df[pd_col].dropna()
        rj_vals = df.loc[pd_vals.index, '_rj']
        if len(pd_vals) > 10 and rj_vals.std() > 0:
            result['rank_correlation'] = pd_vals.corr(rj_vals, method='spearman')

        return result

    def _create_default_proxy(self, df: pd.DataFrame) -> pd.Series:
        """Cria proxy de default a partir de sinais de distress."""
        proxy = pd.Series(0, index=df.index)

        # PL negativo
        for col in ['equity', 'EQUITY']:
            if col in df.columns:
                eq = pd.to_numeric(df[col], errors='coerce')
                proxy = proxy | (eq < 0).fillna(False).astype(int)
                break

        # Altman Z distress
        for col in ['altman_zscore', 'ALTMAN_ZSCORE']:
            if col in df.columns:
                az = pd.to_numeric(df[col], errors='coerce')
                proxy = proxy | (az < 0).fillna(False).astype(int)
                break

        # Status cancelado
        for col in ['SIT_REG', 'is_cancelled']:
            if col in df.columns:
                if col == 'SIT_REG':
                    proxy = proxy | (df[col].str.upper().str.contains(
                        'CANCEL', na=False
                    )).astype(int)
                else:
                    proxy = proxy | df[col].fillna(0).astype(int)
                break

        return proxy

    def _compute_ks(self, pd_vals: pd.Series, default: pd.Series) -> float:
        """Calcula estatística KS (Kolmogorov-Smirnov)."""
        if default.nunique() < 2:
            return 0.0

        df_temp = pd.DataFrame({'pd': pd_vals, 'default': default}).sort_values('pd')
        n_def = default.sum()
        n_nondef = len(default) - n_def

        if n_def == 0 or n_nondef == 0:
            return 0.0

        df_temp['cum_def'] = df_temp['default'].cumsum() / n_def
        df_temp['cum_nondef'] = (1 - df_temp['default']).cumsum() / n_nondef
        df_temp['ks'] = abs(df_temp['cum_def'] - df_temp['cum_nondef'])

        return float(df_temp['ks'].max())

    def _compute_gini(self, pd_vals: pd.Series, default: pd.Series) -> float:
        """Calcula coeficiente de Gini (2*AUC - 1)."""
        if default.nunique() < 2:
            return 0.0

        n_def = default.sum()
        n_nondef = len(default) - n_def

        if n_def == 0 or n_nondef == 0:
            return 0.0

        # Sort by PD descending
        df_temp = pd.DataFrame({'pd': pd_vals, 'default': default})
        df_temp = df_temp.sort_values('pd', ascending=False)

        # Curva CAP (Cumulative Accuracy Profile)
        cum_defaults = df_temp['default'].cumsum()
        total_defaults = cum_defaults.iloc[-1]

        if total_defaults == 0:
            return 0.0

        # AUC via trapézio
        x = np.arange(1, len(df_temp) + 1) / len(df_temp)
        y = cum_defaults.values / total_defaults
        auc = np.trapz(y, x)

        gini = 2 * auc - 1
        return float(np.clip(gini, -1, 1))

    def _decile_analysis(
        self, pd_vals: pd.Series, default: pd.Series
    ) -> pd.DataFrame:
        """Análise de default por decil de PD."""
        if len(pd_vals) < 10:
            return pd.DataFrame()

        df_temp = pd.DataFrame({'pd': pd_vals, 'default': default})
        df_temp['decile'] = pd.qcut(df_temp['pd'], 10, labels=False,
                                     duplicates='drop')

        result = df_temp.groupby('decile').agg(
            n=('pd', 'count'),
            pd_media=('pd', 'mean'),
            n_defaults=('default', 'sum'),
            default_rate_decile=('default', 'mean'),
        ).reset_index()

        result['decile'] = result['decile'] + 1
        return result

    def _empty_result(self) -> BacktestResult:
        return BacktestResult(
            n_total=0, n_defaults=0, n_non_defaults=0,
            default_rate=0, avg_pd_defaults=0, avg_pd_non_defaults=0,
            pd_ratio=0, ks_statistic=0, gini_coefficient=0,
            accuracy_ratio=0, power_curve_area=0.5,
            decile_analysis=pd.DataFrame(), lift_top_decile=0,
        )
