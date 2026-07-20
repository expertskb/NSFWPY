"""
Command Line Interface (CLI) for NSFWPY.
"""
import sys
import json
import click
from pathlib import Path

from .core import load_model
from .constants import DEFAULT_MODEL_PATHS


@click.group()
@click.version_option(version="1.0.0", prog_name="nsfwpy")
def main():
    """NSFWPY CLI - High performance CPU-optimized NSFW image classification engine."""
    pass


@main.command()
@click.argument("image_source", type=str)
@click.option(
    "--model",
    "-m",
    default="nsfw_vit_quantized",
    type=click.Choice(list(DEFAULT_MODEL_PATHS.keys()) + ["custom"]),
    help="Model architecture to use for classification.",
)
@click.option(
    "--model-path",
    type=str,
    default=None,
    help="Custom ONNX model file path if model='custom'.",
)
@click.option(
    "--top-k",
    "-k",
    default=5,
    type=int,
    help="Number of top probability predictions to output.",
)
@click.option("--json-out", is_flag=True, help="Output formatted JSON result.")
def classify(image_source: str, model: str, model_path: str, top_k: int, json_out: bool):
    """Classify an image file path or URL."""
    try:
        chosen_path = model_path if model == "custom" else model
        if not chosen_path:
            click.echo("Error: Please provide --model-path for custom model.", err=True)
            sys.exit(1)

        nsfw_model = load_model(chosen_path)
        predictions = nsfw_model.classify(image_source, top_k=top_k)

        if json_out:
            click.echo(json.dumps(predictions, indent=2))
        else:
            click.echo(f"\nResults for: {image_source}")
            click.echo("=" * 40)
            for p in predictions:
                name = p["className"]
                prob = p["probability"]
                bar = "█" * int(prob * 25)
                click.echo(f"{name:<10} | {prob * 100:>6.2f}% | {bar}")
            click.echo("=" * 40 + "\n")
    except Exception as e:
        click.echo(f"Classification failed: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
