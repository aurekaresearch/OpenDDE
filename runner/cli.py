# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Lightweight console entry point for OpenDDE."""

from __future__ import annotations

import difflib
import importlib
from collections.abc import Mapping

import click

from opendde.version import __version__

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
    "show_default": True,
}
_RUNTIME_COMMAND_MODULE = "runner.batch_inference"
_RUNTIME_COMMANDS = {
    "json": (
        "tojson",
        "Convert PDB or CIF files to OpenDDE inference JSON.",
    ),
    "msa": ("msa", "Run protein MSA search."),
    "mt": ("msatemplate", "Run protein MSA and template search."),
    "pred": ("predict", "Run OpenDDE structure prediction."),
    "prep": (
        "inputprep",
        "Prepare MSA, template, and RNA MSA input features.",
    ),
}
_COMMAND_HELP = {
    "doctor": "Print environment diagnostics and install recommendations.",
    **{name: help_text for name, (_, help_text) in _RUNTIME_COMMANDS.items()},
}


class SuggestGroup(click.Group):
    """A Click group that suggests similar commands on error."""

    def resolve_command(
        self, ctx: click.Context, args: list[str]
    ) -> tuple[str | None, click.Command | None, list[str]]:
        try:
            return super().resolve_command(ctx, args)
        except click.UsageError as exc:
            if args:
                matches = difflib.get_close_matches(args[0], self.list_commands(ctx))
                if matches:
                    suggestion = "\n\nDid you mean one of these?\n    " + ", ".join(
                        matches
                    )
                    raise click.UsageError(
                        f"{exc.message}{suggestion}", ctx=exc.ctx
                    ) from exc
            raise


class LazyOpenDDEGroup(SuggestGroup):
    """Load inference and data dependencies only for commands that need them."""

    def list_commands(self, ctx: click.Context) -> list[str]:
        return sorted(_COMMAND_HELP)

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        command = super().get_command(ctx, cmd_name)
        if command is not None or cmd_name not in _RUNTIME_COMMANDS:
            return command

        attribute_name, _ = _RUNTIME_COMMANDS[cmd_name]
        module = importlib.import_module(_RUNTIME_COMMAND_MODULE)
        command = getattr(module, attribute_name, None)
        if not isinstance(command, click.Command):
            raise TypeError(
                f"Runtime command {attribute_name!r} is not a Click command."
            )
        return command

    def format_commands(
        self, ctx: click.Context, formatter: click.HelpFormatter
    ) -> None:
        rows = [(name, _COMMAND_HELP[name]) for name in self.list_commands(ctx)]
        if rows:
            with formatter.section("Commands"):
                formatter.write_dl(rows)


@click.command(context_settings=CONTEXT_SETTINGS)
def doctor() -> None:
    """Print environment diagnostics and install recommendations."""
    from opendde.utils.environment import format_doctor_report

    click.echo(format_doctor_report())


@click.group(name="opendde", cls=LazyOpenDDEGroup, context_settings=CONTEXT_SETTINGS)
@click.version_option(version=__version__)
def opendde_cli() -> None:
    """OpenDDE: An Open-source Drug Discovery Engine."""


opendde_cli.add_command(doctor)


def register_runtime_commands(namespace: Mapping[str, object]) -> None:
    """Register runtime commands for the legacy ``python -m`` entry point."""
    for public_name, (attribute_name, _) in _RUNTIME_COMMANDS.items():
        command = namespace.get(attribute_name)
        if not isinstance(command, click.Command):
            raise TypeError(
                f"Runtime command {attribute_name!r} is not a Click command."
            )
        opendde_cli.add_command(command, name=public_name)
