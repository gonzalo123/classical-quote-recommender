"""Rich helpers for the terminal interface."""

from __future__ import annotations

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from app.schemas import IngestResponse, QuoteResponse
from settings import APP_ENV

console = Console()


def print_cli_header() -> None:
    """Render a small branded header for the CLI."""

    title = Text("Classical Quote Recommender", style="bold bright_white")
    subtitle = Text(
        f"Rhetorical classical quote recommender  |  env: {APP_ENV}",
        style="cyan",
    )
    console.print(
        Panel(
            Text.assemble(title, "\n", subtitle),
            border_style="blue",
            title="CLI",
            subtitle="Click + Rich",
        )
    )


def print_ingest_result(result: IngestResponse) -> None:
    """Render ingestion results."""

    table = Table(box=box.ROUNDED, header_style="bold magenta")
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    table.add_row("Input path", result.input_path)
    table.add_row("Documents", str(result.documents))
    table.add_row("Chunks", str(result.chunks))
    table.add_row("Embedding backend", result.embedding_backend)
    table.add_row("Vector backend", result.vector_backend)
    table.add_row("Metadata path", result.metadata_path)
    table.add_row("Index path", result.index_path)
    console.print(Panel(table, title="Corpus Indexed", border_style="green"))


def _analysis_table(result: QuoteResponse) -> Table:
    table = Table(box=box.MINIMAL_DOUBLE_HEAD, header_style="bold magenta")
    table.add_column("Signal", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    analysis = result.input_analysis
    table.add_row(
        "Detected language",
        f"{result.detected_language.name} ({result.detected_language.code})",
    )
    table.add_row("Summary", analysis.summary)
    table.add_row("Main theme", analysis.main_theme)
    table.add_row("Secondary themes", ", ".join(analysis.secondary_themes) or "-")
    table.add_row("Tone", analysis.tone)
    table.add_row("Intent", analysis.intent)
    table.add_row("Dominant emotion", analysis.dominant_emotion)
    table.add_row("Quote type", analysis.recommended_quote_type)
    return table


def _quote_panel(title: str, border_style: str, quote) -> Panel:
    body = Text()
    body.append("Original quote\n", style="bold cyan")
    body.append(f"{quote.text}\n\n", style="white")
    if quote.translated_text.strip() != quote.text.strip():
        body.append("Translated quote\n", style="bold green")
        body.append(f"{quote.translated_text}\n\n", style="white")
    body.append(f"{quote.author} · {quote.work} · {quote.reference}\n", style="cyan")
    body.append(f"Score: {quote.score:.4f}\n", style="magenta")
    body.append(f"Why it fits: {quote.why_it_fits}", style="green")
    return Panel(body, title=title, border_style=border_style)


def _alternatives_table(result: QuoteResponse) -> Table:
    show_translation = any(
        alternative.translated_text.strip() != alternative.text.strip()
        for alternative in result.alternatives
    )
    table = Table(box=box.SIMPLE_HEAVY, header_style="bold magenta")
    table.add_column("#", style="cyan", width=3)
    table.add_column("Author / Work", style="white")
    table.add_column("Score", style="magenta", justify="right")
    if show_translation:
        table.add_column("Translated quote", style="white")
    table.add_column("Why it fits", style="green")
    for index, alternative in enumerate(result.alternatives, start=1):
        row = [
            str(index),
            f"{alternative.author} / {alternative.work}",
            f"{alternative.score:.4f}",
        ]
        if show_translation:
            row.append(alternative.translated_text)
        row.append(alternative.why_it_fits)
        table.add_row(*row)
    return table


def print_quote_result(text: str, result: QuoteResponse) -> None:
    """Render a quote recommendation nicely."""

    console.print(Panel(text, title="Input", border_style="blue"))
    console.print(Panel(_analysis_table(result), title="Analysis", border_style="yellow"))
    console.print(_quote_panel("Recommended Quote", "green", result.recommended_quote))
    if result.alternatives:
        console.print(Panel(_alternatives_table(result), title="Alternatives", border_style="magenta"))


def print_demo_header() -> None:
    """Render a section header for demo mode."""

    console.rule("[bold blue]Demo Mode[/bold blue]")
