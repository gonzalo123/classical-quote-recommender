"""Demo command."""

from __future__ import annotations

import click

from app.errors import LocalizationUnavailableError
from app.services.quote_service import QuoteService
from app.utils.rich_cli import console, print_demo_header, print_quote_result
from app.utils.logging import configure_logging
from settings import DATA_PROCESSED_PATH, LOG_LEVEL


@click.command()
def demo() -> None:
    """Run a tiny end-to-end demo."""

    configure_logging(LOG_LEVEL)
    service = QuoteService()
    if not DATA_PROCESSED_PATH.exists():
        with console.status("[bold cyan]Indexing corpus for demo...[/bold cyan]", spinner="dots"):
            service.ingest()

    demo_inputs = [
        "Thanks for your feedback. I do not fully agree with the approach, but I think we can find a middle ground and move forward.",
        "I appreciate the effort. We should keep the discussion focused so the team can act on a shared decision.",
    ]
    print_demo_header()
    for text in demo_inputs:
        try:
            with console.status("[bold cyan]Running demo query...[/bold cyan]", spinner="dots"):
                result = service.quote(text)
        except LocalizationUnavailableError as exc:
            raise click.ClickException(str(exc)) from exc
        print_quote_result(text, result)
        console.rule(style="dim")
