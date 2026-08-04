import logging
from pathlib import Path
from typing import Optional

import click

from colonoscopy_fidelity.io import compute_config, load_cohorts
from colonoscopy_fidelity.network import NetworkShape, model_parameter_count
from colonoscopy_fidelity.pooling import pooled_comparison
from colonoscopy_fidelity.reporting import build_report
from colonoscopy_fidelity.training import configure_logging


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@click.group()
def main() -> None:
    configure_logging()


@main.command("analyze")
@click.option("--root", type=click.Path(path_type=Path), default=None)
@click.option("--output", type=click.Path(path_type=Path), default=None)
def analyze(root: Optional[Path], output: Optional[Path]) -> None:
    selected_root = root if root is not None else project_root()
    selected_output = output if output is not None else selected_root / "outputs"
    build_report(selected_root, selected_output)


@main.command("describe-compute")
@click.option("--root", type=click.Path(path_type=Path), default=None)
def describe_compute(root: Optional[Path]) -> None:
    selected_root = root if root is not None else project_root()
    config = compute_config(selected_root / "configs" / "primary.yaml")
    shape = NetworkShape(hidden_width=config.hidden_width, hidden_depth=config.hidden_depth)
    logging.getLogger(__name__).info(
        "world_size=%d effective_batch=%d parameters=%d epochs=%d learning_rate=%.10g",
        config.world_size,
        config.effective_batch(),
        model_parameter_count(shape),
        config.epochs,
        config.learning_rate,
    )


@main.command("summarize")
@click.option("--root", type=click.Path(path_type=Path), default=None)
def summarize(root: Optional[Path]) -> None:
    selected_root = root if root is not None else project_root()
    cohorts = load_cohorts(selected_root / "data" / "cohorts.csv")
    result = pooled_comparison(cohorts)
    logging.getLogger(__name__).info(
        "implementation=%.3f unsupported=%.3f difference=%.3f odds_ratio=%.3f",
        result.implementation.percent,
        result.unsupported.percent,
        result.difference.value,
        result.odds_ratio.value,
    )


if __name__ == "__main__":
    main()
