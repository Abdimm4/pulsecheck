"""Plot filtered pulse signals and spectra per ROI and print features and the heuristic result.

Usage:
    python scripts/analyze_signals.py path/to/video.mp4
    python scripts/analyze_signals.py path/to/video.mp4 --out data/signal_debug --config my.yaml

Writes signals.png, spectra.png and analysis.json to --out (default
data/signal_debug, which Git ignores). The plots hold colour statistics only,
no face pixels. The heuristic's thresholds are uncalibrated placeholders, so
treat its result as a debugging aid, not as evidence about the video.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

from pulsecheck.config import load_config
from pulsecheck.face import LandmarkerUnavailableError
from pulsecheck.features import QualityConfig, VideoSignals, analyze_traces
from pulsecheck.models import HeuristicConfig, classify
from pulsecheck.signals import SignalConfig, extract_video_traces
from pulsecheck.video import VideoDecodeError

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METHOD_COLORS = {"green": "#2e933c", "chrom": "#7b3fa0"}


def _title(roi: str, analysis: VideoSignals) -> str:
    primary = analysis.signals[roi][analysis.features.pulse_method]
    if not primary.ok:
        return f"{roi} — rejected: {primary.detail}"
    peak = primary.peak
    return (
        f"{roi} — {peak.frequency_hz:.2f} Hz ({60 * peak.frequency_hz:.0f} /min), "
        f"concentration {peak.concentration:.2f}, SNR {peak.snr_db:.1f} dB"
    )


def plot_signals(analysis: VideoSignals, path: Path) -> None:
    rois = list(analysis.signals)
    fig, axes = plt.subplots(len(rois), 1, figsize=(11, 2.6 * len(rois)), sharex=True)
    for ax, roi in zip(axes, rois, strict=True):
        for method, sig in analysis.signals[roi].items():
            if sig.ok:
                ax.plot(sig.t, sig.signal, color=METHOD_COLORS[method], lw=1, label=method)
        ax.set_title(_title(roi, analysis), loc="left", fontsize=10)
        ax.set_ylabel("z-score")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right", ncol=2, frameon=False)
    axes[-1].set_xlabel("time (s)")
    band = analysis.settings["signal"]["band_hz"]
    fig.suptitle(f"Pulse signals, band-passed {band[0]:g}-{band[1]:g} Hz where supported")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_spectra(analysis: VideoSignals, path: Path) -> None:
    rois = list(analysis.signals)
    band = analysis.settings["signal"]["band_hz"]
    fig, axes = plt.subplots(1, len(rois), figsize=(4.2 * len(rois), 3.4), sharey=True)
    for ax, roi in zip(axes, rois, strict=True):
        for method, sig in analysis.signals[roi].items():
            if not sig.ok:
                continue
            spec = sig.spectrum
            shown = spec.freqs_hz <= band[1] * 1.5
            power = spec.power[shown] / spec.power[spec.in_band()].max()
            ax.plot(spec.freqs_hz[shown], power, color=METHOD_COLORS[method], lw=1, label=method)
            ax.axvline(sig.peak.frequency_hz, color=METHOD_COLORS[method], ls=":", lw=1)
        ax.axvspan(*band, color="0.92", zorder=0)
        ax.set_title(roi, loc="left")
        ax.set_xlabel("frequency (Hz)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("power (normalised to in-band max)")
    axes[0].legend(frameon=False)
    fig.suptitle("Welch power spectra (shaded: analysis band, dotted: dominant peak)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/signal_debug"))
    parser.add_argument("--config", type=Path, help="YAML config (default configs/default.yaml)")
    parser.add_argument("--model", type=Path, help="face_landmarker.task path")
    parser.add_argument("--max-frames", type=int, help="only process the first N frames")
    args = parser.parse_args()

    config = load_config(args.config)
    feature_config = config.get("features", {})
    signal_config = SignalConfig.from_mapping(config.get("signal"))
    quality_config = QualityConfig.from_mapping(feature_config.get("quality"))
    heuristic_config = HeuristicConfig.from_mapping(config.get("heuristic"))

    try:
        traces = extract_video_traces(args.video, args.model, max_frames=args.max_frames)
    except (VideoDecodeError, LandmarkerUnavailableError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    analysis = analyze_traces(
        traces, signal_config, feature_config.get("pulse_method", "chrom"), quality_config
    )
    result = classify(analysis.features, heuristic_config)

    args.out.mkdir(parents=True, exist_ok=True)
    plot_signals(analysis, args.out / "signals.png")
    plot_spectra(analysis, args.out / "spectra.png")
    report = {
        "video": str(args.video),
        "tracking": traces.tracking.to_dict(),
        **analysis.summary(),
        "result": result.to_dict(),
    }
    (args.out / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False))

    print("Features (None = missing; reason in parentheses):")
    for name, value in analysis.features.values.items():
        shown = f"{value:.4f}" if value is not None else f"None ({analysis.features.missing[name]})"
        print(f"  {name:42s} {shown}")
    print(f"\nHeuristic result: {result.label}")
    print(
        f"  consistency score: {result.score if result.score is None else round(result.score, 3)}"
    )
    for reason in result.reasons:
        print(f"  - {reason}")
    print("  (uncalibrated thresholds; the score is not a probability)")
    print(f"\nsaved signals.png, spectra.png and analysis.json to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
