import argparse
from typing import List, Optional

import Utils


def parse_csv_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",")]
    return [item for item in items if item]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run TripDNet inference utilities."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    single_parser = subparsers.add_parser(
        "single",
        help="Run TTCDehazeNet on a single hazy image.",
    )
    single_parser.add_argument(
        "--version",
        type=int,
        choices=[1, 2],
        required=True,
        help="TTCDehazeNet version to run.",
    )
    single_parser.add_argument(
        "--hazy-image",
        required=True,
        help="Path to the hazy image.",
    )
    single_parser.add_argument(
        "--gt-image",
        help="Optional ground-truth image path.",
    )
    single_parser.add_argument(
        "--dehazers",
        help="Comma-separated dehazer model names.",
    )
    single_parser.add_argument(
        "--models",
        help="Comma-separated classifier model names (version 1 only).",
    )

    batch_parser = subparsers.add_parser(
        "batch",
        help="Run batch dehaze + evaluate.",
    )
    batch_parser.add_argument(
        "--dehazer",
        required=True,
        help="Dehazer model name (e.g. AllDehazer_LD_40_16_le-4_eph_35).",
    )
    batch_parser.add_argument(
        "--gt-folder",
        required=True,
        help="Folder containing ground-truth images.",
    )
    batch_parser.add_argument(
        "--hazy-folder",
        required=True,
        help="Folder containing hazy images.",
    )

    multi_parser = subparsers.add_parser(
        "multi",
        help="Run haze classification over a folder of images.",
    )
    multi_parser.add_argument(
        "--test-path",
        required=True,
        help="Folder path to run classification inference.",
    )
    multi_parser.add_argument(
        "--models",
        help="Comma-separated classifier model names.",
    )

    return parser


def run_single(args: argparse.Namespace) -> None:
    dehazers = parse_csv_list(args.dehazers)
    model_names = parse_csv_list(args.models) or Utils.selected_models
    Utils.TTCDehazeNet(
        version=args.version,
        gt_image=args.gt_image,
        hazy_image=args.hazy_image,
        model_names=model_names,
        dehazer=dehazers,
    )


def run_batch(args: argparse.Namespace) -> None:
    Utils.batch_dehaze_and_evaluate(
        dehazers=args.dehazer,
        gt_folder=args.gt_folder,
        hazy_folder=args.hazy_folder,
    )


def run_multi(args: argparse.Namespace) -> None:
    model_names = parse_csv_list(args.models) or Utils.selected_models
    Utils.multiple_inference(args.test_path, model_names)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "single":
        run_single(args)
    elif args.command == "batch":
        run_batch(args)
    elif args.command == "multi":
        run_multi(args)


if __name__ == "__main__":
    main()
