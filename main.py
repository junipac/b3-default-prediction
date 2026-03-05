"""
Ponto de entrada CLI do sistema de ingestão B3/CVM.

Uso:
  python main.py daily                    # Pipeline diário
  python main.py quarterly --year 2024   # Pipeline trimestral
  python main.py annual --year 2023      # Pipeline anual completo
  python main.py validate                # Valida disponibilidade das fontes
  python main.py report                  # Relatório de inconsistências
  python main.py reprocess --cnpj 33000167000101
"""

import sys
import json
from datetime import date
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

console = Console()


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Sistema de Ingestão de Dados B3/CVM — Infraestrutura para Modelo de PD."""
    pass


@cli.command()
@click.option("--date", "reference_date", default=None,
              help="Data de referência (YYYY-MM-DD). Default: último dia útil.")
@click.option("--no-db", is_flag=True, default=False,
              help="Executa sem persistência em banco de dados.")
def daily(reference_date: Optional[str], no_db: bool):
    """Pipeline diário: cotações B3 + eventos corporativos."""
    from src.pipeline.orchestrator import PipelineOrchestrator

    ref_date = None
    if reference_date:
        from datetime import datetime
        ref_date = datetime.strptime(reference_date, "%Y-%m-%d").date()

    console.print(Panel(
        f"[bold blue]Pipeline Diário[/bold blue]\n"
        f"Data: {ref_date or 'último dia útil'}\n"
        f"Banco: {'desativado' if no_db else 'ativado'}",
        title="B3 Ingestion"
    ))

    orchestrator = PipelineOrchestrator(use_db=not no_db)
    run = orchestrator.run_daily(reference_date=ref_date)

    _print_run_summary(run)
    sys.exit(0 if run.status.value == "success" else 1)


@cli.command()
@click.option("--year", default=None, type=int,
              help="Ano de referência. Default: todos disponíveis.")
@click.option("--no-db", is_flag=True, default=False)
def quarterly(year: Optional[int], no_db: bool):
    """Pipeline trimestral: ITR da CVM."""
    from src.pipeline.orchestrator import PipelineOrchestrator

    console.print(Panel(
        f"[bold green]Pipeline Trimestral[/bold green]\n"
        f"Ano: {year or 'todos'}",
        title="B3 Ingestion"
    ))

    orchestrator = PipelineOrchestrator(use_db=not no_db)
    run = orchestrator.run_quarterly(year=year)
    _print_run_summary(run)
    sys.exit(0 if run.status.value == "success" else 1)


@cli.command()
@click.option("--year", default=None, type=int,
              help="Ano de referência. Default: todos disponíveis.")
@click.option("--no-db", is_flag=True, default=False)
def annual(year: Optional[int], no_db: bool):
    """Pipeline anual completo: DFP + Cadastro + Detecção de Default."""
    from src.pipeline.orchestrator import PipelineOrchestrator

    console.print(Panel(
        f"[bold yellow]Pipeline Anual[/bold yellow]\n"
        f"Ano: {year or 'todos'}\n"
        f"[italic]Inclui detecção de default e score de distress[/italic]",
        title="B3 Ingestion"
    ))

    orchestrator = PipelineOrchestrator(use_db=not no_db)
    run = orchestrator.run_annual(year=year)
    _print_run_summary(run)
    sys.exit(0 if run.status.value == "success" else 1)


@cli.command()
@click.option("--cnpj", default=None,
              help="CNPJ específico para reprocessamento. Default: todos com pendências.")
@click.option("--no-db", is_flag=True, default=False)
def reprocess(cnpj: Optional[str], no_db: bool):
    """Reprocessa dados revisados/reapresentados."""
    from src.pipeline.orchestrator import PipelineOrchestrator

    console.print(Panel(
        f"[bold magenta]Reprocessamento[/bold magenta]\n"
        f"CNPJ: {cnpj or 'todos com pendências'}",
        title="B3 Ingestion"
    ))

    orchestrator = PipelineOrchestrator(use_db=not no_db)
    run = orchestrator.run_reprocess(cnpj=cnpj)
    _print_run_summary(run)
    sys.exit(0 if run.status.value == "success" else 1)


@cli.command()
def validate():
    """Valida disponibilidade de todas as fontes de dados."""
    from src.pipeline.orchestrator import PipelineOrchestrator

    console.print("[bold]Validando fontes de dados...[/bold]")
    orchestrator = PipelineOrchestrator(use_db=False)
    results = orchestrator.validate_all_sources()

    table = Table(title="Status das Fontes")
    table.add_column("Fonte", style="cyan")
    table.add_column("Status", style="bold")

    for source, ok in results.items():
        status = "[green]✓ Online[/green]" if ok else "[red]✗ Offline[/red]"
        table.add_row(source, status)

    console.print(table)

    all_ok = all(results.values())
    sys.exit(0 if all_ok else 1)


@cli.command()
@click.option("--output", default=None, help="Arquivo de saída CSV.")
def report(output: Optional[str]):
    """Gera relatório consolidado de inconsistências dos últimos runs."""
    from src.pipeline.orchestrator import PipelineOrchestrator

    orchestrator = PipelineOrchestrator(use_db=False)
    df = orchestrator.generate_inconsistency_report()

    if df.empty:
        console.print("[green]Nenhuma inconsistência detectada nos últimos runs.[/green]")
        return

    table = Table(title=f"Relatório de Inconsistências ({len(df)} itens)")
    for col in ["pipeline_type", "severity", "source", "error", "started_at"]:
        if col in df.columns:
            table.add_column(col, max_width=40)

    for _, row in df.head(20).iterrows():
        table.add_row(*[str(row.get(c, ""))[:40] for c in
                        ["pipeline_type", "severity", "source", "error", "started_at"]
                        if c in df.columns])

    console.print(table)

    if output:
        df.to_csv(output, index=False)
        console.print(f"[green]Relatório salvo em: {output}[/green]")


def _print_run_summary(run) -> None:
    """Exibe resumo formatado da execução do pipeline."""
    status_color = {
        "success": "green",
        "partial": "yellow",
        "failed": "red",
        "running": "blue",
    }.get(run.status.value, "white")

    summary = Table(title="Resultado da Execução")
    summary.add_column("Campo")
    summary.add_column("Valor")

    rows = [
        ("Run ID", run.run_id),
        ("Pipeline", run.pipeline_type),
        ("Status", f"[{status_color}]{run.status.value.upper()}[/{status_color}]"),
        ("Registros processados", str(run.records_processed)),
        ("Empresas", str(run.companies_processed)),
        ("Erros", f"[red]{len(run.errors)}[/red]" if run.errors else "0"),
        ("Warnings", f"[yellow]{len(run.warnings)}[/yellow]" if run.warnings else "0"),
        ("Score de qualidade", f"{run.quality_score:.3f}" if run.quality_score else "N/A"),
        ("Início", run.started_at[:19]),
        ("Fim", (run.finished_at or "")[:19]),
    ]

    for field, value in rows:
        summary.add_row(field, value)

    console.print(summary)

    if run.errors:
        console.print("\n[red bold]Erros encontrados:[/red bold]")
        for err in run.errors[:5]:
            console.print(f"  • [{err.get('source', '?')}] {err.get('error', '')[:120]}")


if __name__ == "__main__":
    cli()
