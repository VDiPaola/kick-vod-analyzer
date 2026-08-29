"""Command line interface."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from .classify.pricing import estimate_cost
from .config import load_settings
from .ingest.chat import build_chat_source
from .ingest.vod import resolve_vod
from .pipeline import Pipeline, RunOptions, write_run_report
from .postprocess.outputs import format_duration, format_timestamp

app = typer.Typer(
    add_completion=False,
    help="Detect and segment streamer activity across a Kick VOD.",
)
console = Console()


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=verbose)],
    )


def build_settings(work_dir: Optional[Path], out_dir: Optional[Path]):
    overrides: dict = {}
    if work_dir is not None:
        overrides["work_dir"] = work_dir
    if out_dir is not None:
        overrides["out_dir"] = out_dir
    return load_settings(**overrides)


@app.command()
def analyse(
    url: str = typer.Option(..., "--url", "-u", help="Kick VOD URL."),
    provider: str = typer.Option("gemini", "--provider", "-p", help="gemini, openai, or mock."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override the provider model."),
    mode: str = typer.Option("sync", "--mode", help="sync or batch."),
    chat: str = typer.Option("kick", "--chat", help="Chat source: kick, file, or none."),
    chat_file: Optional[Path] = typer.Option(None, "--chat-file", help="Chat JSON or JSONL input."),
    scene_threshold: Optional[float] = typer.Option(
        None, "--scene-threshold", help="ffmpeg scene score cutoff, 0 to 1."
    ),
    heartbeat: Optional[float] = typer.Option(
        None, "--heartbeat", help="Seconds between fallback checkpoints."
    ),
    max_samples: Optional[int] = typer.Option(
        None, "--max-samples", help="Cap on classified points, 0 for unlimited."
    ),
    work_dir: Optional[Path] = typer.Option(None, "--work-dir"),
    out_dir: Optional[Path] = typer.Option(None, "--out-dir"),
    no_resume: bool = typer.Option(False, "--no-resume", help="Ignore cached scenes and results."),
    keep_frames: bool = typer.Option(False, "--keep-frames", help="Retain extracted burst frames."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Plan and cost only, no API calls."),
    no_wait: bool = typer.Option(False, "--no-wait", help="Submit the batch job and exit."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the full pipeline for one VOD."""
    configure_logging(verbose)
    settings = build_settings(work_dir, out_dir)

    if scene_threshold is not None:
        settings.sampling.scene_threshold = scene_threshold
    if heartbeat is not None:
        settings.sampling.heartbeat_seconds = heartbeat
    if max_samples is not None:
        settings.sampling.max_samples = max_samples

    source = build_chat_source(chat, chat_file=chat_file, timeout=settings.http_timeout)
    pipeline = Pipeline(settings, chat_source=source, progress=_progress)

    options = RunOptions(
        url=url,
        provider=provider,
        model=model,
        mode=mode,
        chat_source_kind=chat,
        chat_file=chat_file,
        resume=not no_resume,
        keep_frames=keep_frames,
        dry_run=dry_run,
        wait_for_batch=not no_wait,
    )
    report = pipeline.run(options)

    if report.vod is not None:
        write_run_report(report, settings.vod_out_dir(report.vod.vod_id) / "run_report.json")

    _render_report(report, dry_run=dry_run)
    if report.errors and not report.timeline:
        raise typer.Exit(code=1)


@app.command()
def estimate(
    url: str = typer.Option(..., "--url", "-u"),
    provider: str = typer.Option("gemini", "--provider", "-p"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    mode: str = typer.Option("batch", "--mode"),
    samples: int = typer.Option(100, "--samples", help="Assumed classification count."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Estimate run cost without touching ffmpeg or any API."""
    configure_logging(verbose)
    settings = build_settings(None, None)
    resolved_model = model or (
        settings.openai_model if provider == "openai" else settings.gemini_model
    )

    try:
        vod = resolve_vod(url, timeout=settings.http_timeout)
        console.print(
            f"[bold]{vod.channel_slug}[/bold] {vod.title or vod.vod_id} "
            f"({format_duration(vod.duration_seconds)})"
        )
    except Exception as exc:
        console.print(f"[yellow]VOD metadata unavailable: {exc}[/yellow]")

    cost = estimate_cost(samples, resolved_model, batch=mode == "batch")
    _render_cost(cost, resolved_model, mode)


@app.command()
def info(
    url: str = typer.Option(..., "--url", "-u"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Resolve a VOD and print its metadata."""
    configure_logging(verbose)
    settings = build_settings(None, None)
    vod = resolve_vod(url, timeout=settings.http_timeout)

    table = Table(show_header=False, box=None)
    table.add_row("VOD id", vod.vod_id)
    table.add_row("Channel", vod.channel_slug)
    table.add_row("Channel id", str(vod.channel_id or "-"))
    table.add_row("Title", vod.title or "-")
    table.add_row("Duration", format_duration(vod.duration_seconds))
    table.add_row("Started", str(vod.started_at_epoch or "-"))
    table.add_row("Playback", (vod.playback_url or "-")[:110])
    console.print(table)


def _progress(stage: str, message: str) -> None:
    console.print(f"[dim]{stage:>8}[/dim]  {message}")


def _render_cost(cost: dict, model: str, mode: str) -> None:
    table = Table(title=f"Estimated cost: {model} ({mode})")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Requests", f"{int(cost['requests'])}")
    table.add_row("Input tokens", f"{int(cost['input_tokens']):,}")
    table.add_row("Output tokens", f"{int(cost['output_tokens']):,}")
    table.add_row("Input cost", f"${cost['input_cost_usd']:.4f}")
    table.add_row("Output cost", f"${cost['output_cost_usd']:.4f}")
    table.add_row("Total", f"${cost['total_cost_usd']:.4f}")
    console.print(table)


def _render_report(report, *, dry_run: bool) -> None:
    if report.cost:
        _render_cost(report.cost, report.timeline.model if report.timeline else "", "run")

    if dry_run:
        console.print(f"[green]Planned {len(report.sample_points)} classification points.[/green]")
        return

    if report.timeline is None:
        for error in report.errors:
            console.print(f"[red]{error}[/red]")
        return

    table = Table(title="Activity timeline")
    table.add_column("Start")
    table.add_column("End")
    table.add_column("Duration", justify="right")
    table.add_column("Activity")
    table.add_column("Conf", justify="right")
    for segment in report.timeline.segments:
        table.add_row(
            format_timestamp(segment.start_seconds),
            format_timestamp(segment.end_seconds),
            format_duration(segment.duration_seconds),
            segment.label,
            f"{segment.confidence_score:.2f}",
        )
    console.print(table)

    for name, path in report.outputs.items():
        console.print(f"[green]{name}[/green] -> {path}")

    if report.errors:
        console.print(f"[yellow]{len(report.errors)} non-fatal issues[/yellow]")
        for error in report.errors[:10]:
            console.print(f"  [yellow]{error}[/yellow]")


if __name__ == "__main__":
    app()
