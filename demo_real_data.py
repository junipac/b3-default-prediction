#!/usr/bin/env python3
"""
DEMONSTRAÇÃO COM DADOS REAIS — B3/CVM
Conecta às fontes oficiais, extrai, processa e apresenta resultados.
"""

import io
import os
import re
import sys
import zipfile
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# ─── Configuração de display ─────────────────────────────────────────────────
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

# ─── Constantes ──────────────────────────────────────────────────────────────
CVM_BASE = "https://dados.cvm.gov.br/dados/CIA_ABERTA"
TIMEOUT = 120

ACCOUNT_MAP = {
    "1": "total_assets",
    "1.01": "current_assets",
    "1.01.01": "cash_equivalents",
    "1.01.03": "receivables",
    "1.01.04": "inventories",
    "1.02": "non_current_assets",
    "1.02.04": "ppe",
    "1.02.05": "intangibles",
    "2": "total_liabilities_equity",
    "2.01": "current_liabilities",
    "2.01.04": "short_term_debt",
    "2.02": "non_current_liabilities",
    "2.02.01": "long_term_debt",
    "2.03": "equity",
    "2.03.01": "paid_in_capital",
    "2.03.09": "retained_earnings",
    "3.01": "net_revenue",
    "3.02": "cogs",
    "3.03": "gross_profit",
    "3.04": "operating_expenses",
    "3.05": "ebit",
    "3.06": "financial_result",
    "3.07": "ebt",
    "3.08": "income_tax",
    "3.11": "net_income",
    "6.01": "cfo",
    "6.02": "cfi",
    "6.03": "cff",
}

# ─── Funções de download ─────────────────────────────────────────────────────

def download(url: str, label: str = "") -> bytes:
    print(f"  ⬇  {label or url.split('/')[-1]} ...", end=" ", flush=True)
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    size_mb = len(r.content) / 1024 / 1024
    print(f"OK ({size_mb:.1f} MB)")
    return r.content


def read_cvm_zip_csv(content: bytes) -> pd.DataFrame:
    frames = []
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".csv"):
                continue
            with zf.open(name) as f:
                raw = f.read()
            for enc in ["latin-1", "utf-8", "cp1252"]:
                try:
                    df = pd.read_csv(
                        io.BytesIO(raw), sep=";", encoding=enc,
                        dtype=str, on_bad_lines="warn", low_memory=False,
                    )
                    if len(df.columns) >= 5:
                        frames.append(df)
                        break
                except Exception:
                    continue
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CADASTRO DE EMPRESAS
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_company_register() -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("1. CADASTRO DE EMPRESAS — CVM (ATIVAS + CANCELADAS)")
    print("=" * 80)

    url = f"{CVM_BASE}/CAD/DADOS/cad_cia_aberta.csv"
    raw = download(url, "Cadastro de Cias Abertas")
    df = pd.read_csv(io.BytesIO(raw), sep=";", encoding="latin-1", dtype=str)

    df["cnpj"] = df["CNPJ_CIA"].apply(lambda x: re.sub(r"\D", "", str(x)).zfill(14))
    df["cnpj_root"] = df["cnpj"].str[:8]
    df["is_active"] = df["SIT"].str.strip().str.upper().isin(["ATIVO"])
    df["is_cancelled"] = df["DT_CANCEL"].notna() & (df["DT_CANCEL"].str.strip() != "")

    total = len(df)
    ativas = df["is_active"].sum()
    canceladas = df["is_cancelled"].sum()

    print(f"\n  Total de empresas na base CVM: {total:,}")
    print(f"  Ativas:                        {ativas:,}")
    print(f"  Canceladas:                    {canceladas:,}")
    print(f"  Outros status:                 {total - ativas - canceladas:,}")

    # Motivos de cancelamento
    if canceladas > 0:
        motivos = (
            df[df["is_cancelled"]]
            .groupby("MOTIVO_CANCEL")
            .size()
            .sort_values(ascending=False)
            .head(10)
        )
        print(f"\n  Top 10 motivos de cancelamento:")
        for motivo, count in motivos.items():
            print(f"    {count:>5,}  {str(motivo)[:60]}")

    # Setores
    setores = df.groupby("SETOR_ATIV").size().sort_values(ascending=False).head(10)
    print(f"\n  Top 10 setores de atuação:")
    for setor, count in setores.items():
        print(f"    {count:>5,}  {str(setor)[:60]}")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DEMONSTRAÇÕES FINANCEIRAS (DFP)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_dfp(year: int) -> Dict[str, pd.DataFrame]:
    print(f"\n{'=' * 80}")
    print(f"2. DEMONSTRAÇÕES FINANCEIRAS — DFP {year} (CONSOLIDADO)")
    print("=" * 80)

    url = f"{CVM_BASE}/DOC/DFP/DADOS/dfp_cia_aberta_{year}.zip"
    raw = download(url, f"DFP {year} (todos os demonstrativos)")

    # Extrair todos os CSVs do ZIP principal
    result = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".csv"):
                continue
            with zf.open(name) as f:
                csv_bytes = f.read()
            df = None
            for enc in ["latin-1", "utf-8"]:
                try:
                    df = pd.read_csv(
                        io.BytesIO(csv_bytes), sep=";", encoding=enc,
                        dtype=str, on_bad_lines="warn", low_memory=False,
                    )
                    break
                except Exception:
                    continue

            if df is None or df.empty:
                continue

            name_upper = name.upper()
            key = None
            if "BPA_CON" in name_upper:
                key = "bpa_con"
            elif "BPP_CON" in name_upper:
                key = "bpp_con"
            elif "DRE_CON" in name_upper:
                key = "dre_con"
            elif "DFC_MI_CON" in name_upper:
                key = "dfc_mi_con"
            elif "DVA_CON" in name_upper:
                key = "dva_con"

            if key:
                result[key] = df
                print(f"  ✓ {key}: {len(df):,} registros, {df['CNPJ_CIA'].nunique() if 'CNPJ_CIA' in df.columns else '?'} empresas")

    return result


def normalize_financial(dfp_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pivota demonstrativos CVM → um registro por empresa com indicadores."""
    frames = []

    for key in ["bpa_con", "bpp_con", "dre_con", "dfc_mi_con"]:
        df = dfp_data.get(key)
        if df is None or df.empty:
            continue

        df = df.copy()
        df["cnpj"] = df["CNPJ_CIA"].apply(lambda x: re.sub(r"\D", "", str(x)).zfill(14))

        # Filtrar apenas exercício atual (ÚLTIMO) e versão mais recente
        if "ORDEM_EXERC" in df.columns:
            df = df[df["ORDEM_EXERC"].str.strip().str.upper() == "ÚLTIMO"]

        if "VERSAO" in df.columns:
            df["VERSAO"] = pd.to_numeric(df["VERSAO"], errors="coerce")
            df = df.sort_values("VERSAO", ascending=False)
            df = df.drop_duplicates(subset=["cnpj", "DT_REFER", "CD_CONTA"], keep="first")

        # Valor numérico
        df["value"] = pd.to_numeric(df["VL_CONTA"].str.replace(",", "."), errors="coerce")

        # Escala
        if "ESCALA_MOEDA" in df.columns:
            scale_map = {"MIL": 1000, "UNIDADE": 1, "MILHAO": 1e6}
            df["scale_mult"] = df["ESCALA_MOEDA"].str.strip().str.upper().map(scale_map).fillna(1000)
            df["value"] = df["value"] * df["scale_mult"]

        # Mapear para contas padrão
        df["account"] = df["CD_CONTA"].map(ACCOUNT_MAP)
        df = df.dropna(subset=["account", "value"])

        pivot = df.pivot_table(
            index=["cnpj", "DT_REFER", "DENOM_CIA"],
            columns="account",
            values="value",
            aggfunc="first",
        ).reset_index()
        pivot.columns.name = None

        frames.append(pivot)

    if not frames:
        return pd.DataFrame()

    # Merge progressivo
    merged = frames[0]
    for f in frames[1:]:
        common_cols = ["cnpj", "DT_REFER", "DENOM_CIA"]
        available = [c for c in common_cols if c in f.columns and c in merged.columns]
        new_cols = [c for c in f.columns if c not in merged.columns or c in available]
        merged = pd.merge(merged, f[new_cols], on=available, how="outer", suffixes=("", "_dup"))

    # Remover duplicadas
    dup_cols = [c for c in merged.columns if c.endswith("_dup")]
    merged = merged.drop(columns=dup_cols)

    return merged


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula indicadores financeiros derivados."""

    def safe_div(n, d):
        return np.where((d != 0) & pd.notna(n) & pd.notna(d), n / d, np.nan)

    # Alavancagem
    if "short_term_debt" in df.columns and "long_term_debt" in df.columns:
        df["total_debt"] = df["short_term_debt"].fillna(0) + df["long_term_debt"].fillna(0)
    if "total_debt" in df.columns and "cash_equivalents" in df.columns:
        df["net_debt"] = df["total_debt"] - df["cash_equivalents"].fillna(0)

    # Liquidez
    if "current_assets" in df.columns and "current_liabilities" in df.columns:
        df["current_ratio"] = safe_div(df["current_assets"], df["current_liabilities"])

    # Endividamento
    if "total_debt" in df.columns and "equity" in df.columns:
        df["debt_to_equity"] = safe_div(df["total_debt"], df["equity"])

    # Dívida Líq / EBIT (proxy EBITDA)
    if "net_debt" in df.columns and "ebit" in df.columns:
        df["net_debt_ebit"] = safe_div(df["net_debt"], df["ebit"])

    # Margens
    if "gross_profit" in df.columns and "net_revenue" in df.columns:
        df["gross_margin"] = safe_div(df["gross_profit"], df["net_revenue"])
    if "ebit" in df.columns and "net_revenue" in df.columns:
        df["ebit_margin"] = safe_div(df["ebit"], df["net_revenue"])
    if "net_income" in df.columns and "net_revenue" in df.columns:
        df["net_margin"] = safe_div(df["net_income"], df["net_revenue"])

    # Rentabilidade
    if "net_income" in df.columns and "equity" in df.columns:
        df["roe"] = safe_div(df["net_income"], df["equity"])
    if "net_income" in df.columns and "total_assets" in df.columns:
        df["roa"] = safe_div(df["net_income"], df["total_assets"])

    # Altman Z-Score (mercados emergentes)
    required_altman = ["current_assets", "current_liabilities", "total_assets",
                       "retained_earnings", "ebit", "equity"]
    if all(c in df.columns for c in required_altman):
        ta = df["total_assets"]
        wc = df["current_assets"] - df["current_liabilities"]
        tl = ta - df["equity"].fillna(0)

        x1 = np.where(ta != 0, wc / ta, np.nan)
        x2 = np.where(ta != 0, df["retained_earnings"].fillna(0) / ta, np.nan)
        x3 = np.where(ta != 0, df["ebit"].fillna(0) / ta, np.nan)
        x4 = np.where((tl != 0) & pd.notna(tl), df["equity"].fillna(0) / tl, np.nan)

        df["altman_zscore"] = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4
        df["altman_zone"] = pd.cut(
            df["altman_zscore"],
            bins=[-np.inf, 1.1, 2.6, np.inf],
            labels=["DISTRESS", "GREY", "SAFE"],
        )

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DATA QUALITY
# ═══════════════════════════════════════════════════════════════════════════════

def run_quality_checks(df: pd.DataFrame) -> Dict:
    print(f"\n{'=' * 80}")
    print("3. DATA QUALITY — VALIDAÇÃO AUTOMÁTICA")
    print("=" * 80)

    results = {"checks": [], "total": len(df)}

    # 3.1 — Completude por campo
    key_fields = ["total_assets", "equity", "net_revenue", "net_income", "ebit",
                  "current_ratio", "debt_to_equity"]
    available = [c for c in key_fields if c in df.columns]
    completeness = df[available].notna().mean()

    print(f"\n  [CHECK 1] Completude por indicador chave:")
    for field, pct in completeness.sort_values(ascending=True).items():
        bar = "█" * int(pct * 30) + "░" * (30 - int(pct * 30))
        color_flag = "✓" if pct >= 0.75 else "⚠" if pct >= 0.50 else "✗"
        print(f"    {color_flag} {field:25s} {bar} {pct*100:5.1f}%")

    results["completeness"] = completeness.to_dict()

    # 3.2 — Balanço equilibrado (Ativo = Passivo + PL)
    if "total_assets" in df.columns and "total_liabilities_equity" in df.columns:
        df_check = df.dropna(subset=["total_assets", "total_liabilities_equity"])
        if not df_check.empty:
            diff_pct = abs(df_check["total_assets"] - df_check["total_liabilities_equity"]) / df_check["total_assets"].replace(0, np.nan)
            balanced = (diff_pct < 0.01).sum()
            total_check = len(df_check)
            print(f"\n  [CHECK 2] Integridade Balanço (Ativo = Passivo + PL):")
            print(f"    Equilibrados (<1% desvio): {balanced:,} / {total_check:,} ({balanced/total_check*100:.1f}%)")
            results["balance_sheet_ok_pct"] = balanced / total_check
    else:
        print(f"\n  [CHECK 2] Integridade Balanço: colunas insuficientes para validação")

    # 3.3 — PL negativo (distress flag)
    if "equity" in df.columns:
        neg_eq = (df["equity"] < 0).sum()
        total_eq = df["equity"].notna().sum()
        print(f"\n  [CHECK 3] Patrimônio Líquido Negativo:")
        print(f"    Empresas com PL < 0:         {neg_eq:,} / {total_eq:,} ({neg_eq/max(total_eq,1)*100:.1f}%)")
        results["negative_equity_count"] = int(neg_eq)

    # 3.4 — Receita negativa
    if "net_revenue" in df.columns:
        neg_rev = (df["net_revenue"] < 0).sum()
        print(f"\n  [CHECK 4] Receita Líquida Negativa:")
        print(f"    Empresas com Receita < 0:    {neg_rev:,}")
        results["negative_revenue_count"] = int(neg_rev)

    # 3.5 — Altman Zone distribution
    if "altman_zone" in df.columns:
        zones = df["altman_zone"].value_counts()
        print(f"\n  [CHECK 5] Distribuição Altman Z-Score (Emergent Markets):")
        for zone, count in zones.items():
            pct = count / len(df) * 100
            marker = {"DISTRESS": "🔴", "GREY": "🟡", "SAFE": "🟢"}.get(str(zone), "⚪")
            bar = "█" * int(pct) + "░" * max(0, 40 - int(pct))
            print(f"    {marker} {str(zone):10s} {bar} {count:>5,} ({pct:5.1f}%)")
        results["altman_distribution"] = zones.to_dict()

    # 3.6 — Score geral
    if available:
        df["completeness_score"] = df[available].notna().sum(axis=1) / len(key_fields)
        avg_score = df["completeness_score"].mean()
        print(f"\n  ── SCORE GERAL DE QUALIDADE: {avg_score*100:.1f}% ──")
        results["overall_score"] = avg_score

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DETECÇÃO DE DEFAULT
# ═══════════════════════════════════════════════════════════════════════════════

def detect_defaults(
    df_financial: pd.DataFrame,
    df_register: pd.DataFrame,
) -> pd.DataFrame:
    print(f"\n{'=' * 80}")
    print("4. DETECÇÃO DE DEFAULT — ANÁLISE MULTI-SINAL")
    print("=" * 80)

    events = []

    # 4.1 — Cancelamentos CVM (default formal)
    cancelled = df_register[df_register["is_cancelled"] == True].copy()
    rj_keywords = ["RECUPERAÇÃO", "RECUPERACAO", "FALÊNCIA", "FALENCIA",
                   "LIQUIDAÇÃO", "LIQUIDACAO", "INSOLVÊN"]

    for _, row in cancelled.iterrows():
        reason = str(row.get("MOTIVO_CANCEL", "")).upper()
        is_default_reason = any(kw in reason for kw in rj_keywords)
        if is_default_reason:
            events.append({
                "cnpj": row["cnpj"],
                "company_name": row["DENOM_SOCIAL"],
                "event_type": "RECUPERACAO_JUDICIAL" if "RECUPERA" in reason else
                              "FALENCIA" if "FALÊN" in reason or "FALEN" in reason else
                              "LIQUIDACAO",
                "event_date": row.get("DT_CANCEL", ""),
                "source": "cvm_cadastro",
                "confidence": 1.0,
            })

    print(f"\n  [SINAL 1] Cancelamentos formais com motivo de default:")
    print(f"    Recuperação Judicial / Falência / Liquidação: {len(events):,}")

    # 4.2 — Distress financeiro (via indicadores)
    distress_from_financials = []
    if "altman_zscore" in df_financial.columns:
        distress_df = df_financial[
            df_financial["altman_zscore"].notna() &
            (df_financial["altman_zscore"] < 1.1)
        ]
        print(f"\n  [SINAL 2] Empresas em zona de distress (Altman Z < 1.1):")
        print(f"    Total: {len(distress_df):,}")

        # Cruzar com PL negativo
        if "equity" in df_financial.columns:
            distress_and_neg_eq = distress_df[distress_df["equity"] < 0]
            print(f"    Com PL negativo simultâneo: {len(distress_and_neg_eq):,}")

            for _, row in distress_and_neg_eq.iterrows():
                distress_from_financials.append({
                    "cnpj": row.get("cnpj", ""),
                    "company_name": row.get("DENOM_CIA", ""),
                    "event_type": "DISTRESS_FINANCEIRO",
                    "event_date": row.get("DT_REFER", ""),
                    "source": "cvm_dfp_indicadores",
                    "confidence": 0.75,
                    "altman_zscore": row.get("altman_zscore"),
                    "equity": row.get("equity"),
                })

    all_events = events + distress_from_financials
    events_df = pd.DataFrame(all_events) if all_events else pd.DataFrame()

    # 4.3 — Resumo consolidado
    print(f"\n  ── TOTAL DE SINAIS DE DEFAULT IDENTIFICADOS: {len(all_events):,} ──")

    if not events_df.empty and "event_type" in events_df.columns:
        by_type = events_df.groupby("event_type").size()
        print(f"\n  Por tipo de evento:")
        for etype, count in by_type.sort_values(ascending=False).items():
            print(f"    {count:>5,}  {etype}")

    return events_df


# ═══════════════════════════════════════════════════════════════════════════════
# 5. APRESENTAÇÃO DO DATASET FINAL
# ═══════════════════════════════════════════════════════════════════════════════

def present_final_dataset(
    df_financial: pd.DataFrame,
    df_register: pd.DataFrame,
    events_df: pd.DataFrame,
):
    print(f"\n{'=' * 80}")
    print("5. DATASET FINAL CONSOLIDADO — FORMATO PARA MODELO DE PD")
    print("=" * 80)

    # Merge financeiro com cadastro
    final = df_financial.copy()

    # Adicionar setor do cadastro
    register_slim = df_register[["cnpj", "SETOR_ATIV", "SIT", "DT_CANCEL", "MOTIVO_CANCEL"]].copy()
    register_slim = register_slim.drop_duplicates(subset=["cnpj"], keep="first")
    final = pd.merge(final, register_slim, on="cnpj", how="left")

    # Adicionar flag de default
    if not events_df.empty:
        default_cnpjs = set(events_df["cnpj"].unique())
        final["default_flag"] = final["cnpj"].isin(default_cnpjs).astype(int)
    else:
        final["default_flag"] = 0

    # Selecionar colunas para apresentação
    display_cols = [
        "DENOM_CIA", "cnpj", "SETOR_ATIV", "DT_REFER",
        "total_assets", "equity", "net_revenue", "net_income", "ebit",
        "current_ratio", "debt_to_equity", "gross_margin", "net_margin",
        "roe", "roa", "altman_zscore", "altman_zone", "default_flag",
    ]
    available_display = [c for c in display_cols if c in final.columns]
    final_display = final[available_display].copy()

    # ─── Estatísticas ─────────────────────────────────────────────────────
    n_total = len(final_display)
    n_companies = final_display["cnpj"].nunique() if "cnpj" in final_display.columns else n_total
    n_defaults = int(final_display["default_flag"].sum()) if "default_flag" in final_display.columns else 0
    default_rate = n_defaults / max(n_companies, 1) * 100

    print(f"\n  ┌────────────────────────────────────────────────────┐")
    print(f"  │  Empresas na base:           {n_companies:>10,}           │")
    print(f"  │  Registros financeiros:      {n_total:>10,}           │")
    print(f"  │  Defaults identificados:     {n_defaults:>10,}           │")
    print(f"  │  Taxa de default:            {default_rate:>9.2f}%           │")
    print(f"  └────────────────────────────────────────────────────┘")

    # ─── Top 20 maiores empresas ──────────────────────────────────────────
    if "total_assets" in final.columns:
        top20 = final.dropna(subset=["total_assets"]).nlargest(20, "total_assets")
        print(f"\n  ── TOP 20 MAIORES EMPRESAS (POR ATIVO TOTAL) ──")
        print(f"  {'Empresa':35s} {'Ativo Total':>18s} {'PL':>15s} {'Receita':>15s} {'LL':>15s} {'Z-Score':>8s} {'Zona':>8s}")
        print(f"  {'─' * 120}")

        for _, row in top20.iterrows():
            name = str(row.get("DENOM_CIA", ""))[:33]
            at = f"R$ {row.get('total_assets', 0)/1e9:,.1f} bi" if pd.notna(row.get("total_assets")) else "N/A"
            eq = f"R$ {row.get('equity', 0)/1e9:,.1f} bi" if pd.notna(row.get("equity")) else "N/A"
            rev = f"R$ {row.get('net_revenue', 0)/1e9:,.1f} bi" if pd.notna(row.get("net_revenue")) else "N/A"
            ni = f"R$ {row.get('net_income', 0)/1e9:,.1f} bi" if pd.notna(row.get("net_income")) else "N/A"
            zscore = f"{row.get('altman_zscore', 0):.2f}" if pd.notna(row.get("altman_zscore")) else "N/A"
            zone = str(row.get("altman_zone", "N/A"))
            print(f"  {name:35s} {at:>18s} {eq:>15s} {rev:>15s} {ni:>15s} {zscore:>8s} {zone:>8s}")

    # ─── Empresas em distress ─────────────────────────────────────────────
    if "altman_zone" in final.columns:
        distress = final[final["altman_zone"] == "DISTRESS"].copy()
        if not distress.empty:
            distress = distress.sort_values("altman_zscore", ascending=True)
            print(f"\n  ── EMPRESAS EM ZONA DE DISTRESS ({len(distress):,}) — TOP 15 PIOR Z-SCORE ──")
            print(f"  {'Empresa':35s} {'Z-Score':>8s} {'PL':>15s} {'LL':>15s} {'Liq.Corr.':>10s} {'D/E':>8s}")
            print(f"  {'─' * 95}")

            for _, row in distress.head(15).iterrows():
                name = str(row.get("DENOM_CIA", ""))[:33]
                z = f"{row.get('altman_zscore', 0):.2f}"
                eq = f"R$ {row.get('equity', 0)/1e6:,.0f} mi" if pd.notna(row.get("equity")) else "N/A"
                ni = f"R$ {row.get('net_income', 0)/1e6:,.0f} mi" if pd.notna(row.get("net_income")) else "N/A"
                cr = f"{row.get('current_ratio', 0):.2f}" if pd.notna(row.get("current_ratio")) else "N/A"
                de = f"{row.get('debt_to_equity', 0):.2f}" if pd.notna(row.get("debt_to_equity")) else "N/A"
                print(f"  {name:35s} {z:>8s} {eq:>15s} {ni:>15s} {cr:>10s} {de:>8s}")

    # ─── Distribuição setorial ────────────────────────────────────────────
    if "SETOR_ATIV" in final.columns:
        setor_stats = final.groupby("SETOR_ATIV").agg(
            empresas=("cnpj", "nunique"),
            ativo_medio=("total_assets", "mean"),
            defaults=("default_flag", "sum"),
        ).sort_values("empresas", ascending=False).head(15)

        print(f"\n  ── DISTRIBUIÇÃO SETORIAL ──")
        print(f"  {'Setor':45s} {'Empresas':>10s} {'Ativo Médio':>18s} {'Defaults':>10s}")
        print(f"  {'─' * 87}")
        for setor, row in setor_stats.iterrows():
            s = str(setor)[:43]
            am = f"R$ {row['ativo_medio']/1e6:,.0f} mi" if pd.notna(row["ativo_medio"]) else "N/A"
            print(f"  {s:45s} {int(row['empresas']):>10,} {am:>18s} {int(row['defaults']):>10,}")

    # ─── Exportar amostra ─────────────────────────────────────────────────
    sample_file = Path("/Users/joaocarlospachecojunior/PD_B3/data/processed/pd_dataset_sample.csv")
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    final_display.to_csv(sample_file, index=False)
    print(f"\n  Amostra exportada: {sample_file}")
    print(f"  Colunas: {', '.join(available_display)}")

    # ─── Resumo estatístico dos indicadores ───────────────────────────────
    numeric_indicators = [
        "total_assets", "equity", "net_revenue", "net_income",
        "current_ratio", "debt_to_equity", "gross_margin", "net_margin",
        "roe", "roa", "altman_zscore",
    ]
    available_num = [c for c in numeric_indicators if c in final.columns]
    if available_num:
        print(f"\n  ── ESTATÍSTICAS DESCRITIVAS DOS INDICADORES ──")
        stats = final[available_num].astype(float).describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
        for col in available_num:
            if col in stats.columns:
                s = stats[col]
                label = col.replace("_", " ").title()[:25]
                print(f"\n    {label}:")
                print(f"      Observações: {int(s['count']):,}")
                print(f"      Média:       {s['mean']:>18,.2f}")
                print(f"      Mediana:     {s['50%']:>18,.2f}")
                print(f"      P5-P95:      [{s['5%']:>15,.2f} , {s['95%']:>15,.2f}]")
                print(f"      Min-Max:     [{s['min']:>15,.2f} , {s['max']:>15,.2f}]")

    return final_display


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║      SISTEMA DE INGESTÃO B3/CVM — DEMONSTRAÇÃO COM DADOS REAIS             ║")
    print("║      Módulo de PD (Probabilidade de Default)                                ║")
    print(f"║      Execução: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):47s}          ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")

    # 1. Cadastro
    df_register = fetch_company_register()

    # 2. DFP (pega os 2 últimos anos disponíveis)
    all_financial = []
    for year in [2023, 2024]:
        try:
            dfp_data = fetch_dfp(year)
            if dfp_data:
                df_norm = normalize_financial(dfp_data)
                if not df_norm.empty:
                    df_with_indicators = compute_indicators(df_norm)
                    all_financial.append(df_with_indicators)
                    print(f"\n  ✓ {year}: {len(df_with_indicators):,} registros processados, "
                          f"{df_with_indicators['cnpj'].nunique()} empresas")
        except Exception as e:
            print(f"\n  ⚠ DFP {year}: {e}")

    if not all_financial:
        print("\nERRO: Nenhum dado financeiro disponível.")
        return

    df_financial = pd.concat(all_financial, ignore_index=True)

    # Manter apenas a observação mais recente por empresa
    df_financial = df_financial.sort_values("DT_REFER", ascending=False)
    df_financial = df_financial.drop_duplicates(subset=["cnpj"], keep="first")

    print(f"\n  ── DADOS FINANCEIROS CONSOLIDADOS ──")
    print(f"    Total de empresas únicas: {df_financial['cnpj'].nunique():,}")
    print(f"    Total de registros:       {len(df_financial):,}")

    # 3. Quality checks
    quality_results = run_quality_checks(df_financial)

    # 4. Default detection
    events_df = detect_defaults(df_financial, df_register)

    # 5. Apresentação final
    final = present_final_dataset(df_financial, df_register, events_df)

    print(f"\n{'=' * 80}")
    print("EXECUÇÃO CONCLUÍDA COM SUCESSO")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
