"""Command line interface for carvllm-trace.

See https://docs.python.org/3/using/cmdline.html#cmdoption-m for why module is
named __main__.py.
"""

from typer import Typer


app = Typer(help="Software for processing of the carvllm-trace")


if __name__ == "__main__":
    app()
