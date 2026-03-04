"""CLI entry point using typer."""

from __future__ import annotations

import asyncio
import sys
from typing import Optional

import typer
from rich.console import Console

from whatllm.hardware.types import HardwareInfo

app = typer.Typer(
    name="llm-checker",
    help="Find the best LLM that runs on your hardware.",
    no_args_is_help=False,
    invoke_without_command=True,
)
console = Console()


def _run_async(coro):
    """Run async coroutine from sync context."""
    return asyncio.run(coro)


def _validate_gpu_flags(
    cpu_only: bool, gpu: str | None, vram: float | None,
) -> None:
    """Validate mutual exclusivity of GPU-related flags."""
    if cpu_only and gpu:
        console.print("[red]Error:[/] --cpu-only and --gpu are mutually exclusive.")
        raise typer.Exit(code=1)
    if vram is not None and not gpu:
        console.print("[red]Error:[/] --vram requires --gpu.")
        raise typer.Exit(code=1)


def _apply_gpu_overrides(
    hardware: HardwareInfo, cpu_only: bool, gpu: str | None, vram: float | None,
) -> HardwareInfo:
    """Replace hardware.gpus based on CLI flags."""
    if cpu_only:
        hardware.gpus = []
    elif gpu:
        from whatllm.hardware.gpu_simulator import create_synthetic_gpu

        try:
            hardware.gpus = [create_synthetic_gpu(gpu, vram)]
        except ValueError as e:
            console.print(f"[red]Error:[/] {e}")
            raise typer.Exit(code=1)
    return hardware


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    refresh: bool = typer.Option(False, "--refresh", help="Ignore cache and re-fetch models"),
    top: int = typer.Option(10, "--top", "-n", help="Number of top models to show"),
    context_length: int = typer.Option(4096, "--context-length", "-c", help="Context length for KV cache estimation"),
    quant: Optional[str] = typer.Option(None, "--quant", "-q", help="Filter by quantization type (e.g. Q4_K_M)"),
    min_speed: Optional[float] = typer.Option(None, "--min-speed", help="Minimum tok/s filter"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    cpu_only: bool = typer.Option(False, "--cpu-only", help="Ignore GPU and run in CPU-only mode"),
    gpu: Optional[str] = typer.Option(None, "--gpu", help="Simulate a GPU (e.g. 'RTX 4090')"),
    vram: Optional[float] = typer.Option(None, "--vram", help="Override VRAM in GB (requires --gpu)"),
):
    """Detect hardware and recommend the best local LLMs."""
    if ctx.invoked_subcommand is not None:
        return

    _validate_gpu_flags(cpu_only, gpu, vram)

    from rich.progress import Progress, SpinnerColumn, TextColumn

    from whatllm.engine.ranker import rank_models
    from whatllm.hardware.detector import detect_hardware
    from whatllm.models.benchmark import (
        fetch_benchmark_scores,
        load_benchmark_cache,
        save_benchmark_cache,
    )
    from whatllm.models.cache import load_cache, save_cache
    from whatllm.models.fetcher import dicts_to_models, fetch_models, models_to_dicts
    from whatllm.models.grouper import group_models
    from whatllm.output.display import display_hardware, display_json, display_ranking

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        # Step 1: Detect hardware
        task = progress.add_task("Detecting hardware...", total=None)
        hardware = detect_hardware()
        _apply_gpu_overrides(hardware, cpu_only, gpu, vram)
        progress.update(task, description="Hardware detected")

        # Step 2: Fetch models
        progress.update(task, description="Loading models...")
        cached_data = None if refresh else load_cache()
        if cached_data is not None:
            models = dicts_to_models(cached_data)
            progress.update(task, description=f"Loaded {len(models)} models from cache")
        else:
            progress.update(task, description="Fetching models from HuggingFace...")
            try:
                models = _run_async(fetch_models())
                save_cache(models_to_dicts(models))
                progress.update(task, description=f"Fetched {len(models)} models")
            except Exception as e:
                console.print(f"[red]Error fetching models:[/] {e}")
                sys.exit(1)

        # Step 3: Fetch benchmark scores
        progress.update(task, description="Loading benchmark data...")
        bench_scores = None if refresh else load_benchmark_cache()
        if bench_scores is None:
            try:
                progress.update(task, description="Fetching benchmark scores...")
                bench_scores = fetch_benchmark_scores()
                save_benchmark_cache(bench_scores)
            except Exception as e:
                console.print(f"[yellow]Warning:[/] Benchmark data unavailable: {e}")
                bench_scores = {}

        # Step 4: Group and rank
        progress.update(task, description="Ranking models...")
        families = group_models(models)

        # Flatten all models with their family IDs set by grouper
        all_models = []
        for family in families:
            all_models.append(family.base_model)
            all_models.extend(family.variants)

        results = rank_models(
            all_models,
            hardware,
            context_length=context_length,
            top_n=top,
            quant_filter=quant,
            min_speed=min_speed,
            benchmark_scores=bench_scores,
        )

    # Display results
    if json_output:
        display_json(results, hardware)
    else:
        console.print()
        display_hardware(hardware)
        console.print()
        display_ranking(results, has_gpu=bool(hardware.gpus))
        console.print()


@app.command()
def hardware(
    cpu_only: bool = typer.Option(False, "--cpu-only", help="Ignore GPU and run in CPU-only mode"),
    gpu: Optional[str] = typer.Option(None, "--gpu", help="Simulate a GPU (e.g. 'RTX 4090')"),
    vram: Optional[float] = typer.Option(None, "--vram", help="Override VRAM in GB (requires --gpu)"),
):
    """Show detected hardware information only."""
    _validate_gpu_flags(cpu_only, gpu, vram)

    from rich.progress import Progress, SpinnerColumn, TextColumn

    from whatllm.hardware.detector import detect_hardware
    from whatllm.output.display import display_hardware

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Detecting hardware...", total=None)
        hw = detect_hardware()
        _apply_gpu_overrides(hw, cpu_only, gpu, vram)
        progress.remove_task(task)

    console.print()
    display_hardware(hw)
    console.print()


if __name__ == "__main__":
    app()
