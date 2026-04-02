"""Click CLI entry point."""

from __future__ import annotations

import click

from commands.demo import demo
from commands.ingest import ingest
from commands.quote import quote
from app.utils.rich_cli import print_cli_header


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Classical quote recommender PoC."""

    if ctx.invoked_subcommand is None:
        print_cli_header()
        click.echo(ctx.get_help())


cli.add_command(ingest)
cli.add_command(quote)
cli.add_command(demo)


if __name__ == "__main__":
    cli()
