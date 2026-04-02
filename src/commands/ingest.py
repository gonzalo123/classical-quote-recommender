"""Ingest command."""

from __future__ import annotations

from pathlib import Path

import click

from app.services.quote_service import QuoteService
from app.utils.rich_cli import console, print_ingest_result
from app.utils.logging import configure_logging
from settings import LOG_LEVEL


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=None,
    help="Corpus directory. Defaults to DATA_RAW_PATH.",
)
def ingest(input_path: Path | None) -> None:
    """Index the local corpus."""

    configure_logging(LOG_LEVEL)
    with console.status("[bold cyan]Indexing corpus...[/bold cyan]", spinner="dots"):
        result = QuoteService().ingest(input_path)
    print_ingest_result(result)
