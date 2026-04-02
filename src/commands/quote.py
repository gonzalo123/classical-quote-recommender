"""Quote command."""

from __future__ import annotations

import click

from app.errors import LocalizationUnavailableError
from app.services.quote_service import QuoteService
from app.utils.rich_cli import console, print_quote_result
from app.utils.logging import configure_logging
from settings import LOG_LEVEL


@click.command()
@click.option("--text", required=True, help="Input email or short text.")
def quote(text: str) -> None:
    """Recommend quotes for an input text."""

    configure_logging(LOG_LEVEL)
    try:
        with console.status("[bold cyan]Analyzing message and selecting quotes...[/bold cyan]", spinner="dots"):
            result = QuoteService().quote(text)
    except LocalizationUnavailableError as exc:
        raise click.ClickException(str(exc)) from exc
    print_quote_result(text, result)
