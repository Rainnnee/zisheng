"""Command-line entry points for training and viewing the 字生 pilot."""

from __future__ import annotations

import argparse
from pathlib import Path
import webbrowser

from .sample import sample
from .train import train_model
from .web import serve


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/runs/pilot"),
        help="directory for checkpoints, figures, and the local demo page",
    )
    parser.add_argument(
        "--font-dir",
        type=Path,
        default=Path("data/fonts"),
        help="directory for the downloaded Noto CJK font files and license texts",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "mps", "cuda", "cpu"),
        help="auto selects Apple MPS, then CUDA, then CPU",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for data and training")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zisheng",
        description="Train and inspect a small class-conditional diffusion pilot.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    demo = commands.add_parser(
        "demo",
        help="train the pilot and create a sample gallery and reverse-process animation",
    )
    _common_arguments(demo)
    demo.add_argument("--steps", type=int, default=4000, help="optimizer updates")
    demo.add_argument("--batch-size", type=int, default=64)
    demo.add_argument("--timesteps", type=int, default=1000)
    demo.add_argument("--variants-per-font", type=int, default=128)
    demo.add_argument("--samples-per-character", type=int, default=4)
    demo.add_argument("--open", action="store_true", help="open the generated report in a browser")

    train = commands.add_parser("train", help="train and save a checkpoint without sampling")
    _common_arguments(train)
    train.add_argument("--steps", type=int, default=4000)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--timesteps", type=int, default=1000)
    train.add_argument("--variants-per-font", type=int, default=128)

    generate = commands.add_parser("sample", help="sample from an existing checkpoint")
    generate.add_argument("--checkpoint", type=Path, required=True)
    generate.add_argument("--output", type=Path, default=Path("results/runs/sampled"))
    generate.add_argument("--font-dir", type=Path, default=Path("data/fonts"))
    generate.add_argument("--device", default="auto", choices=("auto", "mps", "cuda", "cpu"))
    generate.add_argument("--seed", type=int, default=7)
    generate.add_argument("--samples-per-character", type=int, default=4)
    generate.add_argument("--open", action="store_true")

    web = commands.add_parser("web", help="launch the interactive diffusion website on this computer")
    web.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("demo/assets/expanded-12char-48px/zisheng-pilot.pt"),
    )
    web.add_argument("--device", default="auto", choices=("auto", "mps", "cuda", "cpu"))
    web.add_argument("--host", default="127.0.0.1", help="local address; leave at 127.0.0.1 for private use")
    web.add_argument("--port", type=int, default=8765)
    web.add_argument("--no-browser", action="store_true", help="start the server without opening a browser")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.command == "demo":
        checkpoint = train_model(
            arguments.output,
            arguments.font_dir,
            steps=arguments.steps,
            batch_size=arguments.batch_size,
            timesteps=arguments.timesteps,
            variants_per_font=arguments.variants_per_font,
            seed=arguments.seed,
            requested_device=arguments.device,
        )
        page = sample(
            checkpoint,
            seed=arguments.seed + 1,
            output_directory=arguments.output,
            font_directory=arguments.font_dir,
            samples_per_character=arguments.samples_per_character,
            requested_device=arguments.device,
        )
        if arguments.open:
            webbrowser.open(page.resolve().as_uri())
        return

    if arguments.command == "train":
        train_model(
            arguments.output,
            arguments.font_dir,
            steps=arguments.steps,
            batch_size=arguments.batch_size,
            timesteps=arguments.timesteps,
            variants_per_font=arguments.variants_per_font,
            seed=arguments.seed,
            requested_device=arguments.device,
        )
        return

    if arguments.command == "web":
        serve(
            arguments.checkpoint,
            host=arguments.host,
            port=arguments.port,
            requested_device=arguments.device,
            open_browser=not arguments.no_browser,
        )
        return

    page = sample(
        arguments.checkpoint,
        seed=arguments.seed,
        output_directory=arguments.output,
        font_directory=arguments.font_dir,
        samples_per_character=arguments.samples_per_character,
        requested_device=arguments.device,
    )
    if arguments.open:
        webbrowser.open(page.resolve().as_uri())


if __name__ == "__main__":
    main()
