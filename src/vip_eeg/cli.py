import argparse
import json
import logging
from pathlib import Path
from typing import Any

from vip_eeg.config import DEFAULT_CONFIG, load_config
from vip_eeg.data import download_dataset, prepare_topomaps, validate_download
from vip_eeg.evaluation import run_cluster_statistics, run_cnn_loso, run_random_forest
from vip_eeg.plotting import make_paper_figures
from vip_eeg.publication import reproduce_paper_analysis, verify_paper_sources
from vip_eeg.utils import configure_logging
from vip_eeg.verify import verify_installation


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--verbose", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vip-eeg")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download")
    add_common_arguments(download)
    download.add_argument("--validate-only", action="store_true")

    prepare = subparsers.add_parser("prepare")
    add_common_arguments(prepare)
    prepare.add_argument("--recording", action="append")
    prepare.add_argument("--force", action="store_true")

    random_forest = subparsers.add_parser("random-forest")
    add_common_arguments(random_forest)
    random_forest.add_argument("--force", action="store_true")

    statistics = subparsers.add_parser("statistics")
    add_common_arguments(statistics)
    statistics.add_argument("--force", action="store_true")

    cnn = subparsers.add_parser("cnn-loso")
    add_common_arguments(cnn)
    cnn.add_argument("--grouping", choices=["subject_session", "subject"], default=None)
    cnn.add_argument("--fold", action="append")
    cnn.add_argument("--force", action="store_true")

    figures = subparsers.add_parser("figures")
    add_common_arguments(figures)

    analysis = subparsers.add_parser("paper-analysis")
    add_common_arguments(analysis)
    analysis.add_argument("--no-figures", action="store_true")

    verify = subparsers.add_parser("verify")
    add_common_arguments(verify)
    verify.add_argument("--output")

    run_all = subparsers.add_parser("run-all")
    add_common_arguments(run_all)
    run_all.add_argument("--skip-download", action="store_true")
    run_all.add_argument("--skip-random-forest", action="store_true")
    run_all.add_argument("--skip-cnn", action="store_true")
    run_all.add_argument("--strict-participant-loso", action="store_true")
    run_all.add_argument("--force", action="store_true")

    sources = subparsers.add_parser("verify-paper")
    add_common_arguments(sources)
    return parser


def run_complete_pipeline(config: dict[str, Any], arguments: argparse.Namespace) -> dict[str, Any]:
    completed = {}
    if arguments.skip_download:
        completed["download"] = validate_download(config)
    else:
        completed["download"] = download_dataset(config)
    completed["prepare"] = prepare_topomaps(config, force=arguments.force)
    if not arguments.skip_random_forest:
        completed["random_forest"] = run_random_forest(config, force=arguments.force)
    completed["statistics"] = run_cluster_statistics(config, force=arguments.force)
    if not arguments.skip_cnn:
        completed["cnn_subject_session"] = run_cnn_loso(
            config, grouping="subject_session", force=arguments.force
        )
        if arguments.strict_participant_loso:
            completed["cnn_subject"] = run_cnn_loso(
                config, grouping="subject", force=arguments.force
            )
    completed["publication"] = reproduce_paper_analysis(config, figures=True)
    return completed


def dispatch(argv: list[str] | None = None) -> Any:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    configure_logging(arguments.verbose)
    config = load_config(arguments.config)
    logging.info("Loaded configuration %s", Path(arguments.config).resolve())
    if arguments.command == "download":
        return validate_download(config) if arguments.validate_only else download_dataset(config)
    if arguments.command == "prepare":
        return prepare_topomaps(config, arguments.recording, arguments.force)
    if arguments.command == "random-forest":
        return run_random_forest(config, arguments.force)
    if arguments.command == "statistics":
        return run_cluster_statistics(config, arguments.force)
    if arguments.command == "cnn-loso":
        return run_cnn_loso(config, arguments.grouping, arguments.fold, arguments.force)
    if arguments.command == "figures":
        return make_paper_figures(config)
    if arguments.command == "paper-analysis":
        return reproduce_paper_analysis(config, figures=not arguments.no_figures)
    if arguments.command == "verify":
        return verify_installation(config, arguments.output)
    if arguments.command == "verify-paper":
        return verify_paper_sources(config)
    if arguments.command == "run-all":
        return run_complete_pipeline(config, arguments)
    raise ValueError(f"Unknown command: {arguments.command}")


def main(argv: list[str] | None = None) -> None:
    result = dispatch(argv)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
