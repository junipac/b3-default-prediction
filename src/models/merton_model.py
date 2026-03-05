"""
Modelo Estrutural de Merton para Probabilidade de Default.

Implementação baseada no modelo de Robert C. Merton (1974):
- Valor da firma segue processo geométrico browniano: dV = μVdt + σVdW
- Default ocorre quando V(T) < D no horizonte T
- Resolve sistema não-linear para obter valor e volatilidade dos ativos

Referências:
    Merton, R. C. (1974). "On the Pricing of Corporate Debt"
    Crosbie, P. & Bohn, J. (2003). "Modeling Default Risk" (KMV/Moody's)
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import fsolve, brentq
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from enum import Enum
import logging
import warnings

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=RuntimeWarning)


class DriftMode(Enum):
    """Modos de calibração do drift (μ)."""
    RISK_NEUTRAL = "risk_neutral"     # μ = r (risco neutro)
    HISTORICAL = "historical"          # μ = retorno histórico
    CONSERVATIVE = "conservative"      # μ = r - 0.5 * σ²


@dataclass
class MertonInput:
    """Parâmetros de entrada para o modelo de Merton."""
    cnpj: str
    company_name: str
    equity_value: float           # E: valor de mercado do equity
    equity_volatility: float      # σE: volatilidade do equity (anualizada)
    short_term_debt: float        # Dívida de curto prazo
    long_term_debt: float         # Dívida de longo prazo
    risk_free_rate: float         # r: taxa livre de risco (anual)
    time_horizon: float = 1.0    # T: horizonte em anos
    drift_mode: DriftMode = DriftMode.RISK_NEUTRAL
    historical_return: Optional[float] = None  # Retorno histórico (se drift_mode == HISTORICAL)
    sector: Optional[str] = None
    reference_date: Optional[str] = None


@dataclass
class MertonResult:
    """Resultados do modelo de Merton."""
    cnpj: str
    company_name: str
    sector: Optional[str]
    reference_date: Optional[str]

    # Inputs
    equity_value: float
    equity_volatility: float
    debt_barrier: float           # D = CP + 0.5 * LP
    risk_free_rate: float
    time_horizon: float
    drift_mode: str

    # Outputs
    asset_value: float            # V: valor estimado dos ativos
    asset_volatility: float       # σV: volatilidade dos ativos
    drift: float                  # μ: drift utilizado
    distance_to_default: float    # DD
    pd_merton: float              # PD = Φ(-DD)
    leverage_ratio: float         # D/V
    d1: float
    d2: float

    # Convergence
    converged: bool
    iterations: int
    residual: float

    # Rating
    rating_bucket: str = ""

    def __post_init__(self):
        self.rating_bucket = self._assign_rating()

    def _assign_rating(self) -> str:
        pd_val = self.pd_merton
        if pd_val < 0.0003:
            return "AAA"
        elif pd_val < 0.001:
            return "AA"
        elif pd_val < 0.005:
            return "A"
        elif pd_val < 0.02:
            return "BBB"
        elif pd_val < 0.05:
            return "BB"
        elif pd_val < 0.15:
            return "B"
        elif pd_val < 0.35:
            return "CCC"
        elif pd_val < 0.60:
            return "CC"
        else:
            return "D"


class MertonModel:
    """
    Implementação do modelo estrutural de Merton para PD.

    O equity é visto como uma call option europeia sobre os ativos da firma:
        E = V * Φ(d1) - D * exp(-rT) * Φ(d2)

    E a relação entre volatilidades:
        σE = (V / E) * Φ(d1) * σV

    Onde:
        d1 = [ln(V/D) + (r + 0.5σV²)T] / (σV√T)
        d2 = d1 - σV√T

    Distance to Default:
        DD = [ln(V/D) + (μ - 0.5σV²)T] / (σV√T)

    PD = Φ(-DD)
    """

    MAX_ITERATIONS = 200
    TOLERANCE = 1e-8
    MIN_VOLATILITY = 0.01
    MAX_VOLATILITY = 5.0
    MIN_ASSET_VALUE = 1e-6

    def __init__(self, max_iterations: int = 200, tolerance: float = 1e-8):
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def compute_pd(self, inp: MertonInput) -> MertonResult:
        """
        Calcula PD de Merton para uma empresa.

        Resolve sistema não-linear via método iterativo (Newton-Raphson/fsolve).
        """
        # Barreira de default: CP + 0.5 * LP (Moody's KMV convention)
        D = inp.short_term_debt + 0.5 * inp.long_term_debt
        E = inp.equity_value
        sigma_E = inp.equity_volatility
        r = inp.risk_free_rate
        T = inp.time_horizon

        # Validações
        if E <= 0 or D <= 0 or sigma_E <= 0 or T <= 0:
            return self._failed_result(inp, D, "Parâmetros inválidos (E, D, σE ou T <= 0)")

        if np.isnan(E) or np.isnan(D) or np.isnan(sigma_E):
            return self._failed_result(inp, D, "Parâmetros NaN")

        # Drift
        drift = self._compute_drift(inp, sigma_E)

        try:
            V, sigma_V, converged, iterations, residual = self._solve_system(
                E, sigma_E, D, r, T
            )
        except Exception as e:
            logger.warning(f"Merton solver falhou para {inp.company_name}: {e}")
            return self._failed_result(inp, D, str(e))

        if not converged or V <= 0 or sigma_V <= 0:
            # Fallback: usar estimativa simples
            V = E + D
            sigma_V = sigma_E * E / (E + D)
            converged = False
            iterations = self.max_iterations
            residual = float('inf')

        # d1, d2 (para pricing do equity como call)
        sqrt_T = np.sqrt(T)
        d1 = (np.log(V / D) + (r + 0.5 * sigma_V**2) * T) / (sigma_V * sqrt_T)
        d2 = d1 - sigma_V * sqrt_T

        # Distance to Default (usando drift real, não risco-neutro)
        drift_actual = self._compute_drift_for_dd(inp, sigma_V)
        dd = (np.log(V / D) + (drift_actual - 0.5 * sigma_V**2) * T) / (sigma_V * sqrt_T)

        # PD
        pd_merton = float(norm.cdf(-dd))
        pd_merton = np.clip(pd_merton, 1e-8, 1.0 - 1e-8)

        leverage = D / V if V > 0 else float('inf')

        return MertonResult(
            cnpj=inp.cnpj,
            company_name=inp.company_name,
            sector=inp.sector,
            reference_date=inp.reference_date,
            equity_value=E,
            equity_volatility=sigma_E,
            debt_barrier=D,
            risk_free_rate=r,
            time_horizon=T,
            drift_mode=inp.drift_mode.value,
            asset_value=float(V),
            asset_volatility=float(sigma_V),
            drift=float(drift_actual),
            distance_to_default=float(dd),
            pd_merton=float(pd_merton),
            leverage_ratio=float(leverage),
            d1=float(d1),
            d2=float(d2),
            converged=bool(converged),
            iterations=int(iterations),
            residual=float(residual),
        )

    def _solve_system(
        self, E: float, sigma_E: float, D: float, r: float, T: float
    ) -> Tuple[float, float, bool, int, float]:
        """
        Resolve o sistema não-linear de Merton via iteração.

        Sistema:
            f1(V, σV) = V*Φ(d1) - D*exp(-rT)*Φ(d2) - E = 0
            f2(V, σV) = V*Φ(d1)*σV - E*σE = 0

        Método: iteração de ponto fixo (mais estável que Newton-Raphson puro).
        """
        sqrt_T = np.sqrt(T)
        D_disc = D * np.exp(-r * T)

        # Chute inicial: V0 = E + D, σV0 = σE * E / V0
        V = E + D
        sigma_V = sigma_E * E / V

        for iteration in range(self.max_iterations):
            V_old = V
            sigma_V_old = sigma_V

            # d1, d2
            if sigma_V * sqrt_T < 1e-12:
                sigma_V = self.MIN_VOLATILITY
            d1 = (np.log(V / D) + (r + 0.5 * sigma_V**2) * T) / (sigma_V * sqrt_T)
            d2 = d1 - sigma_V * sqrt_T

            Nd1 = norm.cdf(d1)
            Nd2 = norm.cdf(d2)

            # Atualiza σV pela equação de volatilidade
            if Nd1 > 1e-12 and V > 1e-12:
                sigma_V_new = sigma_E * E / (V * Nd1)
            else:
                sigma_V_new = sigma_V

            # Clipping de σV
            sigma_V_new = np.clip(sigma_V_new, self.MIN_VOLATILITY, self.MAX_VOLATILITY)

            # Atualiza d1, d2 com novo σV
            d1 = (np.log(V / D) + (r + 0.5 * sigma_V_new**2) * T) / (sigma_V_new * sqrt_T)
            d2 = d1 - sigma_V_new * sqrt_T
            Nd1 = norm.cdf(d1)
            Nd2 = norm.cdf(d2)

            # Atualiza V pela equação de pricing do equity
            if Nd1 > 1e-12:
                V_new = (E + D_disc * Nd2) / Nd1
            else:
                V_new = E + D

            V_new = max(V_new, self.MIN_ASSET_VALUE)

            # Convergência
            v_change = abs(V_new - V_old) / max(abs(V_old), 1e-12)
            s_change = abs(sigma_V_new - sigma_V_old) / max(abs(sigma_V_old), 1e-12)
            residual = max(v_change, s_change)

            V = V_new
            sigma_V = sigma_V_new

            if residual < self.tolerance:
                return V, sigma_V, True, iteration + 1, residual

        return V, sigma_V, False, self.max_iterations, residual

    def _solve_system_fsolve(
        self, E: float, sigma_E: float, D: float, r: float, T: float
    ) -> Tuple[float, float, bool, int, float]:
        """Alternativa usando scipy.optimize.fsolve (Newton-Raphson)."""
        sqrt_T = np.sqrt(T)
        D_disc = D * np.exp(-r * T)

        def equations(x):
            V, sigma_V = x
            if V <= 0 or sigma_V <= 0:
                return [1e10, 1e10]
            d1 = (np.log(V / D) + (r + 0.5 * sigma_V**2) * T) / (sigma_V * sqrt_T)
            d2 = d1 - sigma_V * sqrt_T
            eq1 = V * norm.cdf(d1) - D_disc * norm.cdf(d2) - E
            eq2 = norm.cdf(d1) * sigma_V * V - sigma_E * E
            return [eq1, eq2]

        V0 = E + D
        sigma_V0 = sigma_E * E / V0

        solution, info, ier, msg = fsolve(
            equations, [V0, sigma_V0], full_output=True
        )

        V, sigma_V = solution
        residual = np.max(np.abs(info['fvec']))
        converged = ier == 1 and V > 0 and sigma_V > 0

        return V, sigma_V, converged, int(info.get('nfev', 0)), residual

    def _compute_drift(self, inp: MertonInput, sigma_default: float) -> float:
        """Calcula drift conforme modo selecionado."""
        if inp.drift_mode == DriftMode.RISK_NEUTRAL:
            return inp.risk_free_rate
        elif inp.drift_mode == DriftMode.HISTORICAL:
            if inp.historical_return is not None:
                return inp.historical_return
            return inp.risk_free_rate
        elif inp.drift_mode == DriftMode.CONSERVATIVE:
            return inp.risk_free_rate - 0.5 * sigma_default**2
        return inp.risk_free_rate

    def _compute_drift_for_dd(self, inp: MertonInput, sigma_V: float) -> float:
        """Calcula drift para Distance to Default."""
        if inp.drift_mode == DriftMode.RISK_NEUTRAL:
            return inp.risk_free_rate
        elif inp.drift_mode == DriftMode.HISTORICAL:
            if inp.historical_return is not None:
                return inp.historical_return
            return inp.risk_free_rate
        elif inp.drift_mode == DriftMode.CONSERVATIVE:
            return inp.risk_free_rate - 0.5 * sigma_V**2
        return inp.risk_free_rate

    def _failed_result(self, inp: MertonInput, D: float, reason: str) -> MertonResult:
        """Retorna resultado com PD = NaN para falhas."""
        logger.debug(f"Merton falhou para {inp.company_name}: {reason}")
        return MertonResult(
            cnpj=inp.cnpj,
            company_name=inp.company_name,
            sector=inp.sector,
            reference_date=inp.reference_date,
            equity_value=inp.equity_value,
            equity_volatility=inp.equity_volatility,
            debt_barrier=D,
            risk_free_rate=inp.risk_free_rate,
            time_horizon=inp.time_horizon,
            drift_mode=inp.drift_mode.value,
            asset_value=float('nan'),
            asset_volatility=float('nan'),
            drift=float('nan'),
            distance_to_default=float('nan'),
            pd_merton=float('nan'),
            leverage_ratio=float('nan'),
            d1=float('nan'),
            d2=float('nan'),
            converged=False,
            iterations=0,
            residual=float('inf'),
        )

    def compute_batch(
        self,
        inputs: List[MertonInput],
        parallel: bool = False,
    ) -> pd.DataFrame:
        """
        Calcula PD de Merton para um lote de empresas.

        Returns:
            DataFrame com todas as colunas de MertonResult.
        """
        results = []
        failed = 0

        for inp in inputs:
            result = self.compute_pd(inp)
            results.append(result)
            if not result.converged:
                failed += 1

        logger.info(
            f"Merton batch: {len(results)} empresas, "
            f"{len(results) - failed} convergidas, {failed} falhas"
        )

        if not results:
            return pd.DataFrame()

        records = []
        for r in results:
            records.append({
                'cnpj': r.cnpj,
                'company_name': r.company_name,
                'sector': r.sector,
                'reference_date': r.reference_date,
                'equity_value': r.equity_value,
                'equity_volatility': r.equity_volatility,
                'debt_barrier': r.debt_barrier,
                'risk_free_rate': r.risk_free_rate,
                'time_horizon': r.time_horizon,
                'drift_mode': r.drift_mode,
                'asset_value': r.asset_value,
                'asset_volatility': r.asset_volatility,
                'drift': r.drift,
                'distance_to_default': r.distance_to_default,
                'pd_merton': r.pd_merton,
                'leverage_ratio': r.leverage_ratio,
                'd1': r.d1,
                'd2': r.d2,
                'converged': r.converged,
                'iterations': r.iterations,
                'residual': r.residual,
                'rating_bucket': r.rating_bucket,
            })

        return pd.DataFrame(records)

    @staticmethod
    def estimate_equity_volatility(
        returns: pd.Series,
        window: int = 252,
        min_periods: int = 60,
    ) -> float:
        """
        Estima volatilidade do equity a partir de retornos diários.

        Args:
            returns: Série de retornos logarítmicos diários.
            window: Janela para cálculo (dias úteis).
            min_periods: Mínimo de observações.

        Returns:
            Volatilidade anualizada.
        """
        if len(returns) < min_periods:
            return float('nan')

        returns_clean = returns.dropna()
        if len(returns_clean) < min_periods:
            return float('nan')

        # Usar últimos 'window' dias
        recent = returns_clean.tail(window)
        daily_vol = recent.std()

        # Anualizar (√252)
        annual_vol = daily_vol * np.sqrt(252)
        return float(annual_vol)

    @staticmethod
    def prepare_inputs_from_fundamentals(
        df: pd.DataFrame,
        risk_free_rate: float = 0.1175,  # SELIC ~11.75%
        drift_mode: DriftMode = DriftMode.RISK_NEUTRAL,
        equity_vol_default: float = 0.45,
    ) -> List[MertonInput]:
        """
        Prepara inputs de Merton a partir de dados fundamentalistas.

        Mapeia colunas do dataset analítico para parâmetros de Merton:
        - equity_value: patrimônio líquido (ou market cap se disponível)
        - short_term_debt: dívida de curto prazo
        - long_term_debt: dívida de longo prazo
        - equity_volatility: estimada a partir de ROA variability ou default

        Args:
            df: DataFrame com colunas do dataset analítico.
            risk_free_rate: Taxa SELIC anual.
            drift_mode: Modo de drift.
            equity_vol_default: Volatilidade default se não disponível.

        Returns:
            Lista de MertonInput.
        """
        inputs = []

        for _, row in df.iterrows():
            cnpj = str(row.get('cnpj_cia', row.get('CNPJ_CIA', '')))
            name = str(row.get('company_name', row.get('DENOM_SOCIAL', '')))
            sector = row.get('sector', row.get('SETOR_ATIV', None))
            ref_date = str(row.get('reference_date', row.get('DT_REFER', '')))

            # Equity value: usar equity (PL) como proxy se market cap indisponível
            equity = row.get('equity', 0)
            if pd.isna(equity) or equity <= 0:
                # Se PL negativo ou ausente, pular
                equity = row.get('total_assets', 0) * 0.3  # proxy conservador
                if pd.isna(equity) or equity <= 0:
                    continue

            # Dívida
            st_debt = row.get('short_term_debt', 0)
            lt_debt = row.get('long_term_debt', 0)
            if pd.isna(st_debt):
                st_debt = row.get('current_liabilities', 0) or 0
            if pd.isna(lt_debt):
                lt_debt = row.get('non_current_liabilities', 0) or 0
            if pd.isna(st_debt):
                st_debt = 0
            if pd.isna(lt_debt):
                lt_debt = 0

            if st_debt + lt_debt <= 0:
                continue

            # Volatilidade do equity
            sigma_E = row.get('equity_volatility', equity_vol_default)
            if pd.isna(sigma_E) or sigma_E <= 0:
                # Heurística: setores mais voláteis
                sigma_E = equity_vol_default

            # ROA como proxy de drift histórico
            historical_ret = row.get('roa', None)
            if pd.notna(historical_ret):
                historical_ret = float(historical_ret)
            else:
                historical_ret = None

            inp = MertonInput(
                cnpj=cnpj,
                company_name=name,
                equity_value=float(equity),
                equity_volatility=float(sigma_E),
                short_term_debt=float(st_debt),
                long_term_debt=float(lt_debt),
                risk_free_rate=risk_free_rate,
                time_horizon=1.0,
                drift_mode=drift_mode,
                historical_return=historical_ret,
                sector=str(sector) if pd.notna(sector) else None,
                reference_date=ref_date,
            )
            inputs.append(inp)

        return inputs
