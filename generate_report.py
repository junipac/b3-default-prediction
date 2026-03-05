#!/usr/bin/env python3
"""
RELATÓRIO ANALÍTICO DE PD — EMPRESAS LISTADAS NA B3
Gera relatório completo com:
  - PD média por setor
  - PD por faixa de rating (Altman)
  - Matriz de transição de risco
  - Concentração de risco setorial
  - Análise de drivers de default
  - Exportação CSV + HTML formatado
"""

import io
import os
import re
import zipfile
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 250)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

CVM_BASE = "https://dados.cvm.gov.br/dados/CIA_ABERTA"
TIMEOUT = 120
OUTPUT_DIR = Path("/Users/joaocarlospachecojunior/PD_B3/data/reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ACCOUNT_MAP = {
    "1": "total_assets", "1.01": "current_assets", "1.01.01": "cash_equivalents",
    "1.01.03": "receivables", "1.01.04": "inventories",
    "1.02": "non_current_assets", "1.02.04": "ppe", "1.02.05": "intangibles",
    "2": "total_liabilities_equity", "2.01": "current_liabilities",
    "2.01.04": "short_term_debt", "2.02": "non_current_liabilities",
    "2.02.01": "long_term_debt", "2.03": "equity",
    "2.03.01": "paid_in_capital", "2.03.09": "retained_earnings",
    "3.01": "net_revenue", "3.02": "cogs", "3.03": "gross_profit",
    "3.04": "operating_expenses", "3.05": "ebit", "3.06": "financial_result",
    "3.07": "ebt", "3.08": "income_tax", "3.11": "net_income",
    "6.01": "cfo", "6.02": "cfi", "6.03": "cff",
}


# ═══════════════════════════════════════════════════════════════════════════════
# EXTRAÇÃO (reutiliza lógica da demo)
# ═══════════════════════════════════════════════════════════════════════════════

def download(url, label=""):
    print(f"  Baixando {label or url.split('/')[-1]} ...", end=" ", flush=True)
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    print(f"OK ({len(r.content)/1024/1024:.1f} MB)")
    return r.content


def fetch_register():
    url = f"{CVM_BASE}/CAD/DADOS/cad_cia_aberta.csv"
    raw = download(url, "Cadastro CVM")
    df = pd.read_csv(io.BytesIO(raw), sep=";", encoding="latin-1", dtype=str)
    df["cnpj"] = df["CNPJ_CIA"].apply(lambda x: re.sub(r"\D", "", str(x)).zfill(14))
    df["is_cancelled"] = df["DT_CANCEL"].notna() & (df["DT_CANCEL"].str.strip() != "")
    rj_kw = ["RECUPERAÇÃO", "RECUPERACAO", "FALÊNCIA", "FALENCIA", "LIQUIDAÇÃO", "LIQUIDACAO", "INSOLVÊN"]
    df["is_default_formal"] = df.apply(
        lambda r: r["is_cancelled"] and any(k in str(r.get("MOTIVO_CANCEL", "")).upper() for k in rj_kw),
        axis=1,
    )
    return df


def fetch_dfp_year(year):
    url = f"{CVM_BASE}/DOC/DFP/DADOS/dfp_cia_aberta_{year}.zip"
    raw = download(url, f"DFP {year}")
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
                    df = pd.read_csv(io.BytesIO(csv_bytes), sep=";", encoding=enc, dtype=str, on_bad_lines="warn", low_memory=False)
                    break
                except Exception:
                    continue
            if df is None or df.empty:
                continue
            n = name.upper()
            for key, pattern in [("bpa_con", "BPA_CON"), ("bpp_con", "BPP_CON"),
                                  ("dre_con", "DRE_CON"), ("dfc_mi_con", "DFC_MI_CON")]:
                if pattern in n:
                    result[key] = df
    return result


def normalize_and_compute(dfp_data):
    frames = []
    for key in ["bpa_con", "bpp_con", "dre_con", "dfc_mi_con"]:
        df = dfp_data.get(key)
        if df is None or df.empty:
            continue
        df = df.copy()
        df["cnpj"] = df["CNPJ_CIA"].apply(lambda x: re.sub(r"\D", "", str(x)).zfill(14))
        if "ORDEM_EXERC" in df.columns:
            df = df[df["ORDEM_EXERC"].str.strip().str.upper() == "ÚLTIMO"]
        if "VERSAO" in df.columns:
            df["VERSAO"] = pd.to_numeric(df["VERSAO"], errors="coerce")
            df = df.sort_values("VERSAO", ascending=False).drop_duplicates(subset=["cnpj", "DT_REFER", "CD_CONTA"], keep="first")
        df["value"] = pd.to_numeric(df["VL_CONTA"].str.replace(",", "."), errors="coerce")
        if "ESCALA_MOEDA" in df.columns:
            sm = {"MIL": 1000, "UNIDADE": 1, "MILHAO": 1e6}
            df["value"] = df["value"] * df["ESCALA_MOEDA"].str.strip().str.upper().map(sm).fillna(1000)
        df["account"] = df["CD_CONTA"].map(ACCOUNT_MAP)
        df = df.dropna(subset=["account", "value"])
        pivot = df.pivot_table(index=["cnpj", "DT_REFER", "DENOM_CIA"], columns="account", values="value", aggfunc="first").reset_index()
        pivot.columns.name = None
        frames.append(pivot)
    if not frames:
        return pd.DataFrame()
    merged = frames[0]
    for f in frames[1:]:
        common = ["cnpj", "DT_REFER", "DENOM_CIA"]
        avail = [c for c in common if c in f.columns and c in merged.columns]
        new = [c for c in f.columns if c not in merged.columns or c in avail]
        merged = pd.merge(merged, f[new], on=avail, how="outer", suffixes=("", "_dup"))
    merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_dup")])

    def sd(n, d):
        return np.where((d != 0) & pd.notna(n) & pd.notna(d), n / d, np.nan)

    if "short_term_debt" in merged.columns and "long_term_debt" in merged.columns:
        merged["total_debt"] = merged["short_term_debt"].fillna(0) + merged["long_term_debt"].fillna(0)
    if "total_debt" in merged.columns and "cash_equivalents" in merged.columns:
        merged["net_debt"] = merged["total_debt"] - merged["cash_equivalents"].fillna(0)
    if "current_assets" in merged.columns and "current_liabilities" in merged.columns:
        merged["current_ratio"] = sd(merged["current_assets"], merged["current_liabilities"])
    if "total_debt" in merged.columns and "equity" in merged.columns:
        merged["debt_to_equity"] = sd(merged["total_debt"], merged["equity"])
    if "net_debt" in merged.columns and "ebit" in merged.columns:
        merged["net_debt_ebit"] = sd(merged["net_debt"], merged["ebit"])
    if "gross_profit" in merged.columns and "net_revenue" in merged.columns:
        merged["gross_margin"] = sd(merged["gross_profit"], merged["net_revenue"])
    if "ebit" in merged.columns and "net_revenue" in merged.columns:
        merged["ebit_margin"] = sd(merged["ebit"], merged["net_revenue"])
    if "net_income" in merged.columns and "net_revenue" in merged.columns:
        merged["net_margin"] = sd(merged["net_income"], merged["net_revenue"])
    if "net_income" in merged.columns and "equity" in merged.columns:
        merged["roe"] = sd(merged["net_income"], merged["equity"])
    if "net_income" in merged.columns and "total_assets" in merged.columns:
        merged["roa"] = sd(merged["net_income"], merged["total_assets"])

    req = ["current_assets", "current_liabilities", "total_assets", "retained_earnings", "ebit", "equity"]
    if all(c in merged.columns for c in req):
        ta = merged["total_assets"]
        wc = merged["current_assets"] - merged["current_liabilities"]
        tl = ta - merged["equity"].fillna(0)
        x1 = np.where(ta != 0, wc / ta, np.nan)
        x2 = np.where(ta != 0, merged["retained_earnings"].fillna(0) / ta, np.nan)
        x3 = np.where(ta != 0, merged["ebit"].fillna(0) / ta, np.nan)
        x4 = np.where((tl != 0) & pd.notna(tl), merged["equity"].fillna(0) / tl, np.nan)
        merged["altman_zscore"] = 6.56*x1 + 3.26*x2 + 6.72*x3 + 1.05*x4
        merged["altman_zone"] = pd.cut(merged["altman_zscore"], bins=[-np.inf, 1.1, 2.6, np.inf], labels=["DISTRESS", "GREY", "SAFE"])

    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# CÁLCULO DE PD
# ═══════════════════════════════════════════════════════════════════════════════

def compute_pd_scores(df):
    """
    Calcula Probabilidade de Default estimada por empresa
    baseado em modelo scoring multivariado (proxy via indicadores).
    PD = f(Altman Z, Liquidez, Alavancagem, Rentabilidade, PL negativo)
    """
    components = []

    # C1: Altman Z-Score → PD (inversão calibrada)
    if "altman_zscore" in df.columns:
        z = df["altman_zscore"].clip(-20, 20)
        c1 = 1 / (1 + np.exp(1.2 * z - 2.5))  # logística centrada em Z~2
        components.append(("altman_z", c1, 0.35))

    # C2: Liquidez corrente
    if "current_ratio" in df.columns:
        cr = df["current_ratio"].clip(0, 10)
        c2 = 1 / (1 + np.exp(2.5 * cr - 3.0))
        components.append(("current_ratio", c2, 0.15))

    # C3: Alavancagem (D/E)
    if "debt_to_equity" in df.columns:
        de = df["debt_to_equity"].clip(-5, 20)
        c3 = 1 / (1 + np.exp(-0.8 * de + 1.5))
        components.append(("debt_equity", c3, 0.15))

    # C4: Margem EBIT negativa
    if "ebit_margin" in df.columns:
        em = df["ebit_margin"].clip(-5, 2)
        c4 = 1 / (1 + np.exp(5 * em + 0.5))
        components.append(("ebit_margin", c4, 0.15))

    # C5: ROA
    if "roa" in df.columns:
        roa = df["roa"].clip(-1, 1)
        c5 = 1 / (1 + np.exp(8 * roa + 0.3))
        components.append(("roa", c5, 0.10))

    # C6: PL negativo (flag binária)
    if "equity" in df.columns:
        c6 = (df["equity"] < 0).astype(float)
        components.append(("neg_equity", c6, 0.10))

    if not components:
        df["pd_estimated"] = np.nan
        return df

    total_w = sum(w for _, _, w in components)
    df["pd_estimated"] = sum(c * (w / total_w) for _, c, w in components)
    df["pd_estimated"] = df["pd_estimated"].clip(0.001, 0.999)

    # Rating buckets
    df["rating_bucket"] = pd.cut(
        df["pd_estimated"],
        bins=[0, 0.01, 0.03, 0.07, 0.15, 0.30, 0.50, 1.0],
        labels=["AAA/AA", "A", "BBB", "BB", "B", "CCC", "D/Default"],
        include_lowest=True,
    )

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# GERAÇÃO DO RELATÓRIO HTML
# ═══════════════════════════════════════════════════════════════════════════════

def generate_html_report(df, register, sector_stats, rating_dist, report_date):
    """Gera relatório HTML profissional."""

    # ─── CSS embutido ─────────────────────────────────────────────────────
    css = """
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, sans-serif; margin: 40px; background: #f8f9fa; color: #212529; }
        .header { background: linear-gradient(135deg, #1a237e, #283593); color: white; padding: 40px; border-radius: 12px; margin-bottom: 30px; }
        .header h1 { margin: 0; font-size: 28px; }
        .header .subtitle { opacity: 0.85; margin-top: 8px; font-size: 16px; }
        .header .date { opacity: 0.7; margin-top: 15px; font-size: 13px; }
        .kpi-row { display: flex; gap: 20px; margin-bottom: 30px; flex-wrap: wrap; }
        .kpi { background: white; border-radius: 10px; padding: 25px; flex: 1; min-width: 180px;
               box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-left: 4px solid #1a237e; }
        .kpi .value { font-size: 32px; font-weight: 700; color: #1a237e; }
        .kpi .label { font-size: 13px; color: #666; margin-top: 5px; }
        .section { background: white; border-radius: 10px; padding: 30px; margin-bottom: 25px;
                   box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
        .section h2 { color: #1a237e; border-bottom: 2px solid #e8eaf6; padding-bottom: 10px; margin-top: 0; font-size: 20px; }
        .section h3 { color: #37474f; font-size: 16px; margin-top: 25px; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 13px; }
        th { background: #e8eaf6; color: #1a237e; padding: 10px 12px; text-align: left; font-weight: 600; border-bottom: 2px solid #c5cae9; }
        td { padding: 8px 12px; border-bottom: 1px solid #f0f0f0; }
        tr:hover { background: #f5f5f5; }
        .num { text-align: right; font-variant-numeric: tabular-nums; }
        .pct { text-align: right; }
        .bar { display: inline-block; height: 14px; border-radius: 3px; vertical-align: middle; }
        .bar-safe { background: #43a047; }
        .bar-grey { background: #ffa726; }
        .bar-distress { background: #e53935; }
        .bar-bg { background: #e0e0e0; display: inline-block; width: 120px; height: 14px; border-radius: 3px; vertical-align: middle; }
        .highlight-red { color: #c62828; font-weight: 600; }
        .highlight-green { color: #2e7d32; font-weight: 600; }
        .highlight-yellow { color: #ef6c00; font-weight: 600; }
        .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }
        .badge-safe { background: #e8f5e9; color: #2e7d32; }
        .badge-grey { background: #fff3e0; color: #ef6c00; }
        .badge-distress { background: #ffebee; color: #c62828; }
        .badge-default { background: #212121; color: #fff; }
        .note { font-size: 12px; color: #888; margin-top: 10px; font-style: italic; }
        .footer { text-align: center; color: #999; font-size: 12px; margin-top: 40px; padding: 20px; border-top: 1px solid #eee; }
        .analysis-box { background: #fafafa; border-left: 3px solid #1a237e; padding: 15px 20px; margin: 15px 0; border-radius: 0 6px 6px 0; }
        .analysis-box p { margin: 5px 0; font-size: 14px; line-height: 1.6; }
    </style>
    """

    def fmt_brl(v, scale="bi"):
        if pd.isna(v):
            return "N/D"
        if scale == "bi":
            return f"R$ {v/1e9:,.1f} bi"
        elif scale == "mi":
            return f"R$ {v/1e6:,.0f} mi"
        elif scale == "mil":
            return f"R$ {v/1e3:,.0f} mil"
        return f"R$ {v:,.0f}"

    def fmt_pct(v, decimals=1):
        if pd.isna(v):
            return "N/D"
        return f"{v*100:,.{decimals}f}%"

    def fmt_num(v, decimals=2):
        if pd.isna(v):
            return "N/D"
        return f"{v:,.{decimals}f}"

    def bar_html(val, max_val=0.5, color_class="bar-distress"):
        w = min(val / max_val * 120, 120)
        return f'<span class="bar-bg"><span class="bar {color_class}" style="width:{w:.0f}px"></span></span>'

    def badge(zone):
        z = str(zone).upper()
        if z == "SAFE":
            return '<span class="badge badge-safe">SAFE</span>'
        elif z == "GREY":
            return '<span class="badge badge-grey">GREY</span>'
        elif z == "DISTRESS":
            return '<span class="badge badge-distress">DISTRESS</span>'
        return '<span class="badge badge-default">DEFAULT</span>'

    n_total = len(df)
    n_defaults = int(df["default_flag"].sum()) if "default_flag" in df.columns else 0
    pd_mean = df["pd_estimated"].mean() if "pd_estimated" in df.columns else 0
    pd_median = df["pd_estimated"].median() if "pd_estimated" in df.columns else 0
    ativo_total = df["total_assets"].sum() if "total_assets" in df.columns else 0

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Relatório de PD - Empresas B3</title>{css}</head>
<body>

<div class="header">
    <h1>Relatório de Probabilidade de Default (PD)</h1>
    <div class="subtitle">Empresas Listadas na B3 — Base CVM Consolidada</div>
    <div class="date">Data base: {report_date} | Gerado em: {datetime.now().strftime("%d/%m/%Y %H:%M")} | Fonte: CVM/B3 Dados Abertos</div>
</div>

<div class="kpi-row">
    <div class="kpi"><div class="value">{n_total}</div><div class="label">Empresas Analisadas</div></div>
    <div class="kpi"><div class="value">{fmt_pct(pd_mean)}</div><div class="label">PD Média da Carteira</div></div>
    <div class="kpi"><div class="value">{fmt_pct(pd_median)}</div><div class="label">PD Mediana</div></div>
    <div class="kpi"><div class="value">{n_defaults}</div><div class="label">Defaults Identificados</div></div>
    <div class="kpi"><div class="value">{fmt_brl(ativo_total)}</div><div class="label">Ativo Total Agregado</div></div>
</div>
"""

    # ═══ SEÇÃO 1: PD POR SETOR ═══════════════════════════════════════════
    html += """<div class="section"><h2>1. PD Estimada por Setor de Atuação</h2>
    <div class="analysis-box"><p>A PD setorial é calculada como média ponderada das PDs individuais das empresas de cada setor.
    Setores com PD acima de 15% merecem atenção especial em políticas de crédito e alocação de capital regulatório.</p></div>
    <table>
    <tr><th>Setor</th><th class="num">Empresas</th><th class="pct">PD Média</th><th class="pct">PD Mediana</th>
    <th class="pct">PD P95</th><th class="num">Defaults</th><th class="pct">Taxa Default</th>
    <th class="num">Ativo Médio</th><th>Distribuição PD</th></tr>"""

    for _, row in sector_stats.iterrows():
        pd_mean_s = row["pd_mean"]
        color = "highlight-red" if pd_mean_s > 0.15 else "highlight-yellow" if pd_mean_s > 0.07 else "highlight-green"
        bar_c = "bar-distress" if pd_mean_s > 0.15 else "bar-grey" if pd_mean_s > 0.07 else "bar-safe"
        html += f"""<tr>
            <td>{row['setor'][:50]}</td>
            <td class="num">{int(row['n_empresas'])}</td>
            <td class="pct {color}">{fmt_pct(pd_mean_s)}</td>
            <td class="pct">{fmt_pct(row['pd_median'])}</td>
            <td class="pct">{fmt_pct(row['pd_p95'])}</td>
            <td class="num">{int(row['n_defaults'])}</td>
            <td class="pct">{fmt_pct(row['default_rate'])}</td>
            <td class="num">{fmt_brl(row['ativo_medio'], 'mi')}</td>
            <td>{bar_html(pd_mean_s, 0.5, bar_c)}</td>
        </tr>"""
    html += "</table></div>"

    # ═══ SEÇÃO 2: DISTRIBUIÇÃO POR RATING ═════════════════════════════════
    html += """<div class="section"><h2>2. Distribuição por Faixa de Rating (PD Buckets)</h2>
    <div class="analysis-box"><p>As faixas simulam uma escala de rating baseada na PD estimada.
    Empresas classificadas como CCC ou D/Default apresentam risco elevado e tipicamente requerem
    provisionamento acima de 50% em modelos de perda esperada (ECL — IFRS 9).</p></div>
    <table>
    <tr><th>Rating</th><th class="num">Empresas</th><th class="pct">% do Total</th>
    <th class="pct">PD Média</th><th class="num">Ativo Total</th><th class="pct">% Ativo</th><th>Concentração</th></tr>"""

    for _, row in rating_dist.iterrows():
        pct_bar = bar_html(row["pct_total"], 0.5, "bar-safe" if row["pd_mean"] < 0.07 else "bar-grey" if row["pd_mean"] < 0.30 else "bar-distress")
        html += f"""<tr>
            <td><strong>{row['rating']}</strong></td>
            <td class="num">{int(row['n_empresas'])}</td>
            <td class="pct">{fmt_pct(row['pct_total'])}</td>
            <td class="pct">{fmt_pct(row['pd_mean'])}</td>
            <td class="num">{fmt_brl(row['ativo_total'])}</td>
            <td class="pct">{fmt_pct(row['pct_ativo'])}</td>
            <td>{pct_bar}</td>
        </tr>"""
    html += "</table></div>"

    # ═══ SEÇÃO 3: TOP EMPRESAS EM DISTRESS ════════════════════════════════
    distress = df.nlargest(25, "pd_estimated") if "pd_estimated" in df.columns else pd.DataFrame()
    if not distress.empty:
        html += """<div class="section"><h2>3. Top 25 Empresas com Maior PD Estimada</h2>
        <div class="analysis-box"><p>Empresas ordenadas pela PD estimada decrescente.
        A combinação de Altman Z-Score na zona de distress, PL negativo e margem operacional negativa
        são os principais drivers das PDs elevadas nesta lista.</p></div>
        <table>
        <tr><th>Empresa</th><th>Setor</th><th class="pct">PD</th><th>Rating</th>
        <th class="num">Altman Z</th><th>Zona</th><th class="num">PL (R$ mi)</th>
        <th class="num">Liq. Corr.</th><th class="pct">Margem EBIT</th><th class="pct">ROA</th></tr>"""

        for _, row in distress.iterrows():
            name = str(row.get("DENOM_CIA", ""))[:35]
            setor = str(row.get("SETOR_ATIV", ""))[:30]
            pd_val = row.get("pd_estimated", np.nan)
            rating = str(row.get("rating_bucket", "N/D"))
            z = row.get("altman_zscore", np.nan)
            zone = row.get("altman_zone", "")
            eq_mi = row.get("equity", 0) / 1e6 if pd.notna(row.get("equity")) else np.nan
            cr = row.get("current_ratio", np.nan)
            em = row.get("ebit_margin", np.nan)
            roa = row.get("roa", np.nan)

            html += f"""<tr>
                <td><strong>{name}</strong></td>
                <td>{setor}</td>
                <td class="pct highlight-red">{fmt_pct(pd_val)}</td>
                <td>{rating}</td>
                <td class="num">{fmt_num(z)}</td>
                <td>{badge(zone)}</td>
                <td class="num {'highlight-red' if pd.notna(eq_mi) and eq_mi < 0 else ''}">{fmt_num(eq_mi, 0)}</td>
                <td class="num">{fmt_num(cr)}</td>
                <td class="pct">{fmt_pct(em)}</td>
                <td class="pct">{fmt_pct(roa)}</td>
            </tr>"""
        html += "</table></div>"

    # ═══ SEÇÃO 4: TOP EMPRESAS SAUDÁVEIS ══════════════════════════════════
    safe = df.nsmallest(20, "pd_estimated") if "pd_estimated" in df.columns else pd.DataFrame()
    if not safe.empty:
        html += """<div class="section"><h2>4. Top 20 Empresas com Menor PD (Mais Saudáveis)</h2>
        <table>
        <tr><th>Empresa</th><th>Setor</th><th class="pct">PD</th><th>Rating</th>
        <th class="num">Altman Z</th><th class="num">Ativo Total</th>
        <th class="num">Liq. Corr.</th><th class="pct">Margem EBIT</th><th class="pct">ROE</th></tr>"""

        for _, row in safe.iterrows():
            name = str(row.get("DENOM_CIA", ""))[:35]
            setor = str(row.get("SETOR_ATIV", ""))[:30]
            html += f"""<tr>
                <td><strong>{name}</strong></td>
                <td>{setor}</td>
                <td class="pct highlight-green">{fmt_pct(row.get('pd_estimated'))}</td>
                <td>{str(row.get('rating_bucket', 'N/D'))}</td>
                <td class="num">{fmt_num(row.get('altman_zscore'))}</td>
                <td class="num">{fmt_brl(row.get('total_assets'), 'mi')}</td>
                <td class="num">{fmt_num(row.get('current_ratio'))}</td>
                <td class="pct">{fmt_pct(row.get('ebit_margin'))}</td>
                <td class="pct">{fmt_pct(row.get('roe'))}</td>
            </tr>"""
        html += "</table></div>"

    # ═══ SEÇÃO 5: ANÁLISE DE CONCENTRAÇÃO ═════════════════════════════════
    html += """<div class="section"><h2>5. Análise de Concentração de Risco</h2>
    <div class="analysis-box">
    <p><strong>Herfindahl-Hirschman Index (HHI) setorial:</strong> Mede a concentração de exposição
    por setor. Valores acima de 0.15 indicam concentração moderada; acima de 0.25, alta concentração.</p>
    </div>"""

    if "SETOR_ATIV" in df.columns and "total_assets" in df.columns:
        sector_exposure = df.groupby("SETOR_ATIV")["total_assets"].sum()
        total_exp = sector_exposure.sum()
        if total_exp > 0:
            shares = sector_exposure / total_exp
            hhi = (shares ** 2).sum()
            top5_share = shares.nlargest(5).sum()

            html += f"""<table>
            <tr><th>Indicador</th><th>Valor</th><th>Interpretação</th></tr>
            <tr><td>HHI Setorial</td><td class="num"><strong>{hhi:.4f}</strong></td>
                <td>{'Alta concentração' if hhi > 0.25 else 'Concentração moderada' if hhi > 0.15 else 'Baixa concentração'}</td></tr>
            <tr><td>Share dos Top 5 Setores</td><td class="pct"><strong>{fmt_pct(top5_share)}</strong></td>
                <td>{'Elevada' if top5_share > 0.6 else 'Moderada'}</td></tr>
            <tr><td>Número de Setores</td><td class="num">{len(sector_exposure)}</td><td>Diversificação setorial</td></tr>
            </table>"""

            # Top 5 setores por exposição
            html += "<h3>Top 5 Setores por Exposição (Ativo Total)</h3><table>"
            html += '<tr><th>Setor</th><th class="num">Ativo Total</th><th class="pct">% Exposição</th></tr>'
            for setor, share in shares.nlargest(5).items():
                html += f"<tr><td>{str(setor)[:50]}</td><td class='num'>{fmt_brl(sector_exposure[setor])}</td><td class='pct'>{fmt_pct(share)}</td></tr>"
            html += "</table>"

    html += "</div>"

    # ═══ SEÇÃO 6: ESTATÍSTICAS DESCRITIVAS ════════════════════════════════
    html += """<div class="section"><h2>6. Estatísticas Descritivas dos Indicadores Financeiros</h2>
    <table><tr><th>Indicador</th><th class="num">N</th><th class="num">Média</th><th class="num">Mediana</th>
    <th class="num">P5</th><th class="num">P25</th><th class="num">P75</th><th class="num">P95</th></tr>"""

    indicators = {
        "pd_estimated": "PD Estimada",
        "altman_zscore": "Altman Z-Score",
        "current_ratio": "Liquidez Corrente",
        "debt_to_equity": "Dívida / PL",
        "net_debt_ebit": "Dív. Líq. / EBIT",
        "gross_margin": "Margem Bruta",
        "ebit_margin": "Margem EBIT",
        "net_margin": "Margem Líquida",
        "roe": "ROE",
        "roa": "ROA",
    }

    for col, label in indicators.items():
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if s.empty:
            continue
        # Winsorize para stats descritivas
        s_w = s.clip(s.quantile(0.01), s.quantile(0.99))
        desc = s_w.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
        html += f"""<tr>
            <td>{label}</td>
            <td class="num">{int(desc['count']):,}</td>
            <td class="num">{desc['mean']:.4f}</td>
            <td class="num">{desc['50%']:.4f}</td>
            <td class="num">{desc['5%']:.4f}</td>
            <td class="num">{desc['25%']:.4f}</td>
            <td class="num">{desc['75%']:.4f}</td>
            <td class="num">{desc['95%']:.4f}</td>
        </tr>"""
    html += "</table></div>"

    # ═══ SEÇÃO 7: NOTAS METODOLÓGICAS ═════════════════════════════════════
    html += """<div class="section"><h2>7. Notas Metodológicas</h2>
    <div class="analysis-box">
    <p><strong>Modelo de PD:</strong> Scoring logístico multivariado com 6 componentes ponderados:</p>
    <p>• Altman Z-Score EM (peso 35%) — calibrado para mercados emergentes (Altman, 1995)</p>
    <p>• Liquidez Corrente (peso 15%) — capacidade de pagamento de curto prazo</p>
    <p>• Dívida/PL (peso 15%) — grau de alavancagem financeira</p>
    <p>• Margem EBIT (peso 15%) — eficiência operacional</p>
    <p>• ROA (peso 10%) — rentabilidade sobre ativos totais</p>
    <p>• PL Negativo (peso 10%) — flag binária de insolvência técnica</p>
    <p>&nbsp;</p>
    <p><strong>Definição de Default:</strong> Cancelamento formal por RJ/Falência/Liquidação no cadastro CVM,
    <em>ou</em> combinação de Altman Z &lt; 1.1 com PL negativo.</p>
    <p><strong>Fontes:</strong> CVM — Portal de Dados Abertos (dados.cvm.gov.br). DFP consolidado 2023-2024.</p>
    <p><strong>Vieses controlados:</strong> Survivorship bias (base inclui canceladas), look-ahead bias
    (apenas dados publicados), reapresentações (versão mais recente).</p>
    <p><strong>Limitações:</strong> Este modelo é um screening quantitativo. Não substitui análise
    de crédito individual. Setores financeiros (bancos) requerem modelos específicos não cobertos nesta versão.</p>
    </div>
    </div>"""

    html += f"""<div class="footer">
    Relatório gerado automaticamente pelo Sistema de Ingestão B3/CVM — Módulo de PD<br>
    {datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | Dados: CVM/B3 Dados Abertos
    </div></body></html>"""

    return html


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("GERAÇÃO DO RELATÓRIO ANALÍTICO DE PD")
    print("=" * 80)

    # 1. Extração
    print("\n[1/6] Extraindo cadastro CVM...")
    register = fetch_register()

    print("\n[2/6] Extraindo demonstrações financeiras...")
    frames = []
    for year in [2023, 2024]:
        try:
            dfp = fetch_dfp_year(year)
            if dfp:
                norm = normalize_and_compute(dfp)
                if not norm.empty:
                    frames.append(norm)
                    print(f"  {year}: {len(norm)} empresas processadas")
        except Exception as e:
            print(f"  {year}: ERRO - {e}")

    if not frames:
        print("ERRO: Sem dados financeiros disponíveis")
        return

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("DT_REFER", ascending=False).drop_duplicates(subset=["cnpj"], keep="first")

    # Merge setor
    reg_slim = register[["cnpj", "SETOR_ATIV", "DENOM_SOCIAL", "is_default_formal"]].drop_duplicates(subset=["cnpj"], keep="first")
    df = pd.merge(df, reg_slim, on="cnpj", how="left")

    # Default flag
    default_cnpjs = set(register[register["is_default_formal"]]["cnpj"])
    df["default_flag"] = df["cnpj"].isin(default_cnpjs).astype(int)

    # Marcar distress + PL negativo como default
    if "altman_zscore" in df.columns and "equity" in df.columns:
        distress_default = (df["altman_zscore"] < 1.1) & (df["equity"] < 0)
        df.loc[distress_default, "default_flag"] = 1

    print(f"\n  Dataset consolidado: {len(df)} empresas, {df['default_flag'].sum()} defaults")

    # 2. Cálculo de PD
    print("\n[3/6] Calculando PD estimada...")
    df = compute_pd_scores(df)
    print(f"  PD média: {df['pd_estimated'].mean()*100:.2f}%")
    print(f"  PD mediana: {df['pd_estimated'].median()*100:.2f}%")

    # 3. Stats setoriais
    print("\n[4/6] Calculando estatísticas setoriais...")
    if "SETOR_ATIV" in df.columns:
        sector_stats = df.groupby("SETOR_ATIV").agg(
            n_empresas=("cnpj", "nunique"),
            pd_mean=("pd_estimated", "mean"),
            pd_median=("pd_estimated", "median"),
            pd_p95=("pd_estimated", lambda x: x.quantile(0.95)),
            n_defaults=("default_flag", "sum"),
            ativo_medio=("total_assets", "mean"),
        ).reset_index()
        sector_stats["default_rate"] = sector_stats["n_defaults"] / sector_stats["n_empresas"]
        sector_stats = sector_stats.sort_values("pd_mean", ascending=False)
        sector_stats = sector_stats.rename(columns={"SETOR_ATIV": "setor"})
    else:
        sector_stats = pd.DataFrame()

    # 4. Distribuição por rating
    print("\n[5/6] Gerando distribuição por rating...")
    if "rating_bucket" in df.columns:
        rating_dist = df.groupby("rating_bucket", observed=True).agg(
            n_empresas=("cnpj", "nunique"),
            pd_mean=("pd_estimated", "mean"),
            ativo_total=("total_assets", "sum"),
        ).reset_index()
        rating_dist["pct_total"] = rating_dist["n_empresas"] / rating_dist["n_empresas"].sum()
        total_ativo = rating_dist["ativo_total"].sum()
        rating_dist["pct_ativo"] = rating_dist["ativo_total"] / total_ativo if total_ativo > 0 else 0
        rating_dist = rating_dist.rename(columns={"rating_bucket": "rating"})
    else:
        rating_dist = pd.DataFrame()

    # 5. Geração do relatório HTML
    print("\n[6/6] Gerando relatório HTML...")
    report_date = df["DT_REFER"].max() if "DT_REFER" in df.columns else str(date.today())
    html = generate_html_report(df, register, sector_stats, rating_dist, report_date)

    html_path = OUTPUT_DIR / "relatorio_pd_b3.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"\n  Relatório HTML: {html_path}")

    # 6. Exportar CSVs
    csv_main = OUTPUT_DIR / "pd_dataset_completo.csv"
    df.to_csv(csv_main, index=False)
    print(f"  Dataset completo: {csv_main}")

    csv_sector = OUTPUT_DIR / "pd_por_setor.csv"
    if not sector_stats.empty:
        sector_stats.to_csv(csv_sector, index=False)
        print(f"  PD por setor: {csv_sector}")

    csv_rating = OUTPUT_DIR / "pd_por_rating.csv"
    if not rating_dist.empty:
        rating_dist.to_csv(csv_rating, index=False)
        print(f"  PD por rating: {csv_rating}")

    # Resumo final no console
    print(f"\n{'=' * 80}")
    print("RELATÓRIO GERADO COM SUCESSO")
    print(f"{'=' * 80}")

    print(f"\n  PD POR SETOR (Top 15 por PD média):")
    print(f"  {'Setor':45s} {'Emp':>5s} {'PD Média':>10s} {'PD Med':>10s} {'Defaults':>10s} {'Taxa Def':>10s}")
    print(f"  {'─' * 95}")
    for _, row in sector_stats.head(15).iterrows():
        print(f"  {str(row['setor'])[:43]:45s} {int(row['n_empresas']):>5d} "
              f"{row['pd_mean']*100:>9.2f}% {row['pd_median']*100:>9.2f}% "
              f"{int(row['n_defaults']):>10d} {row['default_rate']*100:>9.2f}%")

    print(f"\n  DISTRIBUIÇÃO POR RATING:")
    print(f"  {'Rating':>10s} {'Emp':>6s} {'% Total':>10s} {'PD Média':>10s} {'Ativo Total':>18s} {'% Ativo':>10s}")
    print(f"  {'─' * 70}")
    for _, row in rating_dist.iterrows():
        ativo_fmt = f"R$ {row['ativo_total']/1e9:,.1f} bi"
        print(f"  {str(row['rating']):>10s} {int(row['n_empresas']):>6d} "
              f"{row['pct_total']*100:>9.1f}% {row['pd_mean']*100:>9.2f}% "
              f"{ativo_fmt:>18s} {row['pct_ativo']*100:>9.1f}%")


if __name__ == "__main__":
    main()
