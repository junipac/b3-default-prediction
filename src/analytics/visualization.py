"""
Módulo de Visualização para Análise de PD e Desempenho Setorial.

Gera gráficos embeddable em HTML (base64 PNG via matplotlib):
    - Gráficos de linha por setor (índice, SHI, PD)
    - Heatmap intersetorial
    - Dashboard executivo
    - Distribuições de PD
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import io
import base64
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Estilo global
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafafa',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'figure.titlesize': 14,
})

# Paleta de cores profissional
COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
]


def fig_to_base64(fig: plt.Figure, dpi: int = 120) -> str:
    """Converte figura matplotlib para string base64."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def plot_pd_by_sector(
    sector_agg: pd.DataFrame,
    top_n: int = 20,
    title: str = "PD Merton Média por Setor",
) -> str:
    """
    Gráfico de barras horizontais da PD média por setor.

    Returns:
        String base64 da imagem PNG.
    """
    if sector_agg.empty:
        return ""

    df = sector_agg.nlargest(top_n, 'pd_media').copy()
    df = df.sort_values('pd_media', ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.4)))

    # Cores baseadas em PD
    colors = []
    for pd_val in df['pd_media']:
        if pd_val < 0.02:
            colors.append('#4caf50')
        elif pd_val < 0.05:
            colors.append('#8bc34a')
        elif pd_val < 0.15:
            colors.append('#ff9800')
        elif pd_val < 0.35:
            colors.append('#f44336')
        else:
            colors.append('#b71c1c')

    bars = ax.barh(range(len(df)), df['pd_media'] * 100, color=colors,
                   edgecolor='white', linewidth=0.5, height=0.7)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['sector'].str[:35], fontsize=9)
    ax.set_xlabel('PD Média (%)')
    ax.set_title(title, fontweight='bold', pad=15)

    # Labels nas barras
    for i, (bar, pd_val, n) in enumerate(
        zip(bars, df['pd_media'], df['n_empresas'])
    ):
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f'{pd_val*100:.1f}% (n={int(n)})',
            va='center', ha='left', fontsize=8, color='#333'
        )

    ax.set_xlim(0, df['pd_media'].max() * 100 * 1.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    return fig_to_base64(fig)


def plot_dd_distribution(
    df: pd.DataFrame,
    dd_col: str = 'distance_to_default',
    title: str = "Distribuição da Distance-to-Default",
) -> str:
    """Histograma da distance-to-default."""
    if df.empty or dd_col not in df.columns:
        return ""

    dd = df[dd_col].dropna()
    dd = dd[dd.between(-5, 15)]

    if len(dd) < 10:
        return ""

    fig, ax = plt.subplots(figsize=(10, 6))

    n_bins = min(50, max(20, len(dd) // 10))
    n, bins, patches = ax.hist(
        dd, bins=n_bins, color='#1f77b4', alpha=0.7,
        edgecolor='white', linewidth=0.5
    )

    # Colorir por zona
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge < 1.1:
            patch.set_facecolor('#d32f2f')  # Distress
        elif left_edge < 2.6:
            patch.set_facecolor('#ff9800')  # Grey zone
        else:
            patch.set_facecolor('#4caf50')  # Safe

    # Linhas verticais de referência
    ax.axvline(x=1.1, color='red', linestyle='--', alpha=0.7, label='Zona Distress (DD=1.1)')
    ax.axvline(x=2.6, color='orange', linestyle='--', alpha=0.7, label='Zona Grey (DD=2.6)')

    # Estatísticas
    stats_text = (
        f'Média: {dd.mean():.2f}\n'
        f'Mediana: {dd.median():.2f}\n'
        f'Desvio: {dd.std():.2f}\n'
        f'N: {len(dd)}'
    )
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Distance-to-Default')
    ax.set_ylabel('Frequência')
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    return fig_to_base64(fig)


def plot_pd_distribution(
    df: pd.DataFrame,
    pd_col: str = 'pd_merton',
    title: str = "Distribuição da PD (Merton)",
) -> str:
    """Histograma da PD."""
    if df.empty or pd_col not in df.columns:
        return ""

    pd_vals = df[pd_col].dropna()
    pd_vals = pd_vals[pd_vals.between(0, 1)]

    if len(pd_vals) < 10:
        return ""

    fig, ax = plt.subplots(figsize=(10, 6))

    # Log-scale para melhor visualização
    pd_pct = pd_vals * 100
    n_bins = min(50, max(20, len(pd_pct) // 10))
    ax.hist(pd_pct, bins=n_bins, color='#1f77b4', alpha=0.7,
            edgecolor='white', linewidth=0.5)

    # Rating thresholds
    thresholds = [
        (0.03, 'AAA', '#2e7d32'),
        (0.1, 'AA', '#4caf50'),
        (0.5, 'A', '#8bc34a'),
        (2.0, 'BBB', '#cddc39'),
        (5.0, 'BB', '#ffeb3b'),
        (15.0, 'B', '#ff9800'),
        (35.0, 'CCC', '#f44336'),
    ]
    for thresh, label, color in thresholds:
        ax.axvline(x=thresh, color=color, linestyle=':', alpha=0.6, linewidth=1)

    stats_text = (
        f'Média: {pd_pct.mean():.2f}%\n'
        f'Mediana: {pd_pct.median():.2f}%\n'
        f'P90: {pd_pct.quantile(0.9):.2f}%\n'
        f'N: {len(pd_pct)}'
    )
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('PD (%)')
    ax.set_ylabel('Frequência')
    ax.set_title(title, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    return fig_to_base64(fig)


def plot_sector_heatmap(
    sector_agg: pd.DataFrame,
    metrics: List[str] = None,
    title: str = "Heatmap de Indicadores Setoriais",
) -> str:
    """
    Heatmap de indicadores por setor (normalizado por coluna).
    """
    if sector_agg.empty:
        return ""

    if metrics is None:
        metrics = [
            'pd_media', 'pd_mediana', 'pd_p90',
            'pct_investment_grade', 'pct_distress',
        ]

    available = [m for m in metrics if m in sector_agg.columns]
    if not available:
        return ""

    # Pegar top 15 setores por número de empresas
    top = sector_agg.nlargest(15, 'n_empresas')
    data = top.set_index('sector')[available].copy()

    # Normalizar colunas para [0, 1]
    for col in data.columns:
        vmin, vmax = data[col].min(), data[col].max()
        if vmax > vmin:
            data[col] = (data[col] - vmin) / (vmax - vmin)

    fig, ax = plt.subplots(figsize=(12, max(6, len(data) * 0.5)))

    # Colormap
    cmap = LinearSegmentedColormap.from_list(
        'risk', ['#4caf50', '#ffeb3b', '#ff9800', '#f44336'], N=256
    )

    im = ax.imshow(data.values, cmap=cmap, aspect='auto')

    ax.set_xticks(range(len(available)))
    col_labels = {
        'pd_media': 'PD Média',
        'pd_mediana': 'PD Mediana',
        'pd_p90': 'PD P90',
        'pct_investment_grade': '% Inv Grade',
        'pct_distress': '% Distress',
        'dd_media': 'DD Média',
        'pd_ponderada': 'PD Pond.',
    }
    ax.set_xticklabels(
        [col_labels.get(m, m) for m in available],
        rotation=45, ha='right', fontsize=9
    )

    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data.index.str[:30], fontsize=9)

    # Valores no heatmap
    for i in range(len(data)):
        for j in range(len(available)):
            val = top.iloc[i][available[j]]
            if available[j].startswith('pct_'):
                text = f'{val*100:.0f}%'
            elif available[j].startswith('pd_'):
                text = f'{val*100:.1f}%'
            else:
                text = f'{val:.2f}'
            ax.text(j, i, text, ha='center', va='center',
                    fontsize=8, color='black')

    ax.set_title(title, fontweight='bold', pad=15)
    plt.colorbar(im, ax=ax, label='Normalizado [0-1]', shrink=0.8)
    fig.tight_layout()

    return fig_to_base64(fig)


def plot_shi_chart(
    shi_df: pd.DataFrame,
    title: str = "Sectoral Health Index (SHI)",
) -> str:
    """Gráfico de barras do SHI por setor."""
    if shi_df.empty or 'shi_normalized' not in shi_df.columns:
        return ""

    df = shi_df.sort_values('shi_normalized').copy()

    fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.45)))

    colors = df['color'].tolist() if 'color' in df.columns else ['#1f77b4'] * len(df)

    bars = ax.barh(range(len(df)), df['shi_normalized'], color=colors,
                   edgecolor='white', linewidth=0.5, height=0.7)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['sector'].str[:35], fontsize=9)
    ax.set_xlabel('SHI (Sectoral Health Index)')
    ax.set_title(title, fontweight='bold', pad=15)

    # Referência
    ax.axvline(x=0, color='#333', linewidth=1, linestyle='-')
    ax.axvline(x=-1, color='red', linewidth=0.5, linestyle=':', alpha=0.5)
    ax.axvline(x=1, color='green', linewidth=0.5, linestyle=':', alpha=0.5)

    # Labels
    for i, (bar, shi_val, cls) in enumerate(
        zip(bars, df['shi_normalized'], df['classification'])
    ):
        offset = 0.05 if shi_val >= 0 else -0.05
        ha = 'left' if shi_val >= 0 else 'right'
        ax.text(
            shi_val + offset, i,
            f'{shi_val:+.2f} ({cls})',
            va='center', ha=ha, fontsize=8, color='#333'
        )

    xlim = max(abs(df['shi_normalized'].min()), abs(df['shi_normalized'].max())) + 1
    ax.set_xlim(-xlim, xlim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    return fig_to_base64(fig)


def plot_rating_distribution(
    rating_dist: pd.DataFrame,
    title: str = "Distribuição por Rating (Merton)",
) -> str:
    """Gráfico de barras da distribuição por rating."""
    if rating_dist.empty:
        return ""

    fig, ax = plt.subplots(figsize=(10, 6))

    rating_colors = {
        'AAA': '#1b5e20', 'AA': '#2e7d32', 'A': '#4caf50',
        'BBB': '#8bc34a', 'BB': '#cddc39', 'B': '#ff9800',
        'CCC': '#f44336', 'CC': '#c62828', 'D': '#b71c1c',
    }

    colors = [rating_colors.get(r, '#9e9e9e') for r in rating_dist['rating']]

    bars = ax.bar(range(len(rating_dist)), rating_dist['pct'],
                  color=colors, edgecolor='white', linewidth=0.5)

    ax.set_xticks(range(len(rating_dist)))
    ax.set_xticklabels(rating_dist['rating'], fontsize=11, fontweight='bold')

    # Labels nas barras
    for bar, pct, n in zip(bars, rating_dist['pct'], rating_dist['n']):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{pct:.1f}%\n(n={int(n)})',
            ha='center', va='bottom', fontsize=9
        )

    ax.set_ylabel('% do Total')
    ax.set_title(title, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    return fig_to_base64(fig)


def plot_merton_scatter(
    df: pd.DataFrame,
    x_col: str = 'leverage_ratio',
    y_col: str = 'distance_to_default',
    pd_col: str = 'pd_merton',
    title: str = "Alavancagem vs Distance-to-Default",
) -> str:
    """Scatter plot leverage vs DD, colorido por PD."""
    if df.empty:
        return ""

    valid = df.dropna(subset=[x_col, y_col, pd_col])
    valid = valid[valid[x_col].between(0, 3)]
    valid = valid[valid[y_col].between(-3, 15)]

    if len(valid) < 10:
        return ""

    fig, ax = plt.subplots(figsize=(10, 7))

    scatter = ax.scatter(
        valid[x_col], valid[y_col],
        c=valid[pd_col] * 100,
        cmap='RdYlGn_r',
        s=30, alpha=0.6, edgecolors='white', linewidth=0.3
    )

    plt.colorbar(scatter, ax=ax, label='PD (%)', shrink=0.8)

    # Zonas de referência
    ax.axhline(y=1.1, color='red', linestyle='--', alpha=0.4, label='Zona Distress')
    ax.axhline(y=2.6, color='orange', linestyle='--', alpha=0.4, label='Zona Grey')

    ax.set_xlabel('Leverage Ratio (D/V)')
    ax.set_ylabel('Distance-to-Default')
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    return fig_to_base64(fig)


def plot_convergence_stats(
    df: pd.DataFrame,
    title: str = "Convergência do Solver de Merton",
) -> str:
    """Estatísticas de convergência do solver."""
    if df.empty or 'converged' not in df.columns:
        return ""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pie chart convergência
    conv = df['converged'].value_counts()
    labels = ['Convergiu', 'Não convergiu']
    sizes = [conv.get(True, 0), conv.get(False, 0)]
    colors_pie = ['#4caf50', '#f44336']
    axes[0].pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 10})
    axes[0].set_title('Taxa de Convergência', fontweight='bold')

    # Histograma de iterações
    iters = df[df['converged']]['iterations']
    if len(iters) > 0:
        axes[1].hist(iters, bins=20, color='#1f77b4', alpha=0.7,
                     edgecolor='white')
        axes[1].set_xlabel('Iterações até Convergência')
        axes[1].set_ylabel('Frequência')
        axes[1].set_title('Distribuição de Iterações', fontweight='bold')
        axes[1].axvline(x=iters.median(), color='red', linestyle='--',
                       label=f'Mediana: {iters.median():.0f}')
        axes[1].legend()

    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    fig.suptitle(title, fontweight='bold', fontsize=14, y=1.02)
    fig.tight_layout()

    return fig_to_base64(fig)


def plot_temporal_pd_variation(
    temporal_df: pd.DataFrame,
    top_n: int = 20,
    title: str = "Variação da PD Merton por Setor (2023 → 2024)",
) -> str:
    """
    Gráfico butterfly: setores que melhoraram (verde) vs pioraram (vermelho).
    """
    if temporal_df.empty or 'delta_pd' not in temporal_df.columns:
        return ""

    df = temporal_df.dropna(subset=['delta_pd']).copy()
    df = df.sort_values('delta_pd')

    if len(df) > top_n:
        top_pos = df.nlargest(top_n // 2, 'delta_pd')
        top_neg = df.nsmallest(top_n // 2, 'delta_pd')
        df = pd.concat([top_neg, top_pos]).drop_duplicates()
        df = df.sort_values('delta_pd')

    fig, ax = plt.subplots(figsize=(13, max(6, len(df) * 0.45)))

    colors = ['#4caf50' if d < 0 else '#f44336' for d in df['delta_pd']]

    bars = ax.barh(range(len(df)), df['delta_pd'] * 100, color=colors,
                   edgecolor='white', linewidth=0.5, height=0.7)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['sector'].str[:35], fontsize=9)
    ax.set_xlabel('Δ PD (pontos percentuais)')
    ax.set_title(title, fontweight='bold', pad=15)
    ax.axvline(x=0, color='#333', linewidth=1)

    for i, (bar, delta) in enumerate(zip(bars, df['delta_pd'])):
        pd23 = df.iloc[i].get('pd_2023', 0) if 'pd_2023' in df.columns else 0
        pd24 = df.iloc[i].get('pd_2024', 0) if 'pd_2024' in df.columns else 0
        offset = 0.1 if delta >= 0 else -0.1
        ha = 'left' if delta >= 0 else 'right'
        label = f'{delta*100:+.2f}pp'
        if pd23 > 0:
            label += f' ({pd23*100:.1f}%→{pd24*100:.1f}%)'
        ax.text(delta * 100 + offset, i, label,
                va='center', ha=ha, fontsize=8, color='#333')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    return fig_to_base64(fig)


def plot_temporal_indicators(
    temporal_df: pd.DataFrame,
    title: str = "Evolução de Indicadores por Setor (2023 → 2024)",
) -> str:
    """Gráfico comparativo multi-indicador."""
    if temporal_df.empty:
        return ""

    metrics = []
    for col in ['delta_roa', 'delta_leverage', 'delta_altman', 'delta_dd']:
        if col in temporal_df.columns:
            metrics.append(col)

    if not metrics:
        return ""

    df = temporal_df.nlargest(15, 'n_common')

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 8))
    if len(metrics) == 1:
        axes = [axes]

    metric_labels = {
        'delta_roa': ('Δ ROA (pp)', True),
        'delta_leverage': ('Δ D/E', False),
        'delta_altman': ('Δ Altman Z', True),
        'delta_dd': ('Δ Distance-to-Default', True),
    }

    for ax, metric in zip(axes, metrics):
        label, positive_is_good = metric_labels.get(metric, (metric, True))
        vals = df[metric].fillna(0)
        if metric == 'delta_roa':
            vals = vals * 100

        colors = []
        for v in vals:
            if positive_is_good:
                colors.append('#4caf50' if v > 0 else '#f44336')
            else:
                colors.append('#f44336' if v > 0 else '#4caf50')

        ax.barh(range(len(df)), vals, color=colors,
                edgecolor='white', linewidth=0.5, height=0.6)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df['sector'].str[:25], fontsize=8)
        ax.set_xlabel(label, fontsize=9)
        ax.axvline(x=0, color='#333', linewidth=0.8)
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle(title, fontweight='bold', fontsize=13, y=1.02)
    fig.tight_layout()

    return fig_to_base64(fig)


def plot_sector_migration(
    temporal_df: pd.DataFrame,
    title: str = "Migração de PD Setorial (2023 → 2024)",
) -> str:
    """Scatter PD 2023 vs PD 2024 com linha de 45° em escala logarítmica.

    Utiliza escala log-log para lidar com a ampla faixa de PDs
    (tipicamente 0.001% a >40%), evitando compressão dos pontos de
    baixa PD que constituem a maioria das observações.
    """
    if temporal_df.empty:
        return ""

    needed = ['pd_2023', 'pd_2024', 'sector']
    if not all(c in temporal_df.columns for c in needed):
        return ""

    df = temporal_df.dropna(subset=['pd_2023', 'pd_2024']).copy()
    if len(df) < 3:
        return ""

    fig, ax = plt.subplots(figsize=(11, 10))

    # Converter para % e garantir mínimo para escala log (0.001%)
    MIN_PD_PCT = 0.001
    df['_x'] = (df['pd_2023'] * 100).clip(lower=MIN_PD_PCT)
    df['_y'] = (df['pd_2024'] * 100).clip(lower=MIN_PD_PCT)

    # Tamanho proporcional ao número de empresas
    sizes = df.get('n_common', pd.Series(10, index=df.index))
    sizes = (sizes / sizes.max() * 300).clip(30, 500)

    # Cores: vermelho se piorou, verde se melhorou
    colors = ['#d32f2f' if row['_y'] > row['_x'] * 1.05 else
              '#2e7d32' if row['_y'] < row['_x'] * 0.95 else
              '#ff9800'  # laranja se variação < 5%
              for _, row in df.iterrows()]

    ax.scatter(df['_x'], df['_y'], s=sizes, c=colors, alpha=0.65,
               edgecolors='white', linewidth=0.5, zorder=3)

    # Escala logarítmica em ambos os eixos
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Limites dos eixos
    all_vals = pd.concat([df['_x'], df['_y']])
    lo = max(MIN_PD_PCT * 0.5, all_vals.min() * 0.5)
    hi = all_vals.max() * 2.0
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    # Linha de 45° (sem variação) — reta em log-log
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.35, linewidth=1.2,
            label='Sem variação', zorder=1)

    # Faixas de deterioração e melhoria (shading em log)
    ax.fill_between([lo, hi], [lo, hi], [hi, hi],
                    alpha=0.04, color='red', label='Deterioração', zorder=0)
    ax.fill_between([lo, hi], [lo, lo], [lo, hi],
                    alpha=0.04, color='green', label='Melhoria', zorder=0)

    # Linhas de referência para ratings
    rating_thresholds = [
        (0.5, 'A/BBB', '#8bc34a'),
        (2.0, 'BBB/BB', '#cddc39'),
        (5.0, 'BB/B', '#ff9800'),
        (15.0, 'B/CCC', '#f44336'),
    ]
    for thresh, label, color in rating_thresholds:
        if lo < thresh < hi:
            ax.axhline(y=thresh, color=color, linestyle=':', alpha=0.3, linewidth=0.8)
            ax.axvline(x=thresh, color=color, linestyle=':', alpha=0.3, linewidth=0.8)
            ax.text(hi * 0.85, thresh * 1.15, label, fontsize=6.5,
                    color=color, alpha=0.7, ha='right')

    # Formatação dos ticks para mostrar %
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f'{v:.2f}%' if v < 0.1 else (f'{v:.1f}%' if v < 10 else f'{v:.0f}%')
        ))

    # Anotações: usar repel simples para evitar sobreposição
    texts = []
    for _, row in df.iterrows():
        sector_label = str(row['sector'])[:22]
        delta = (row['_y'] - row['_x'])
        delta_pct = delta / max(row['_x'], 0.001) * 100
        detail = f'{row["_x"]:.2f}%→{row["_y"]:.2f}%'

        # Anotar setores relevantes (grandes variações ou PD alto)
        is_notable = (abs(delta_pct) > 30 or row['_y'] > 5 or row['_x'] > 5
                      or (row.get('n_common', 0) > 5))

        if is_notable and len(texts) < 25:
            ax.annotate(
                f'{sector_label}\n({detail})',
                (row['_x'], row['_y']),
                fontsize=6.5, alpha=0.85,
                xytext=(6, 6), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='#ccc', alpha=0.85),
            )
            texts.append(1)

    # Legenda
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d32f2f',
               markersize=8, label='Piora (↑PD >5%)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff9800',
               markersize=8, label='Estável (±5%)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2e7d32',
               markersize=8, label='Melhora (↓PD >5%)'),
        Line2D([0], [0], linestyle='--', color='k', alpha=0.4,
               label='Sem variação'),
    ]
    ax.legend(handles=legend_elements, fontsize=8.5, loc='lower right',
              framealpha=0.9)

    # Nota sobre escala
    ax.text(
        0.02, 0.98,
        'Escala logarítmica · Tamanho ∝ nº empresas no setor',
        transform=ax.transAxes, fontsize=7.5, va='top', ha='left',
        color='#888',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    )

    ax.set_xlabel('PD Média 2023 (%)', fontsize=11)
    ax.set_ylabel('PD Média 2024 (%)', fontsize=11)
    ax.set_title(title, fontweight='bold', pad=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, which='both', alpha=0.2, linewidth=0.5)
    fig.tight_layout()

    return fig_to_base64(fig)
