"""Audit a feature table for source/compression confounds (shortcut check).

Usage:
    python scripts/audit_confounds.py data/features/nfi.csv \
        --media data/features/nfi_media.csv --manifest data/manifests/nfi.csv
    python scripts/audit_confounds.py data/features/ffpp_c23.csv \
        --media data/features/ffpp_c23_media.csv --manifest data/manifests/ffpp_c23.csv

On the train split (by default) it reports (a) the single-feature ROC-AUC,
real vs fake, of every media/shortcut column and (b) for every pulse
classifier feature its Spearman correlation with each media column and the
grouped cross-validated R^2 of predicting it from all media columns, flagging
features with |rho| >= 0.3 or R^2 >= 0.2 (pulsecheck.dataset.confounds).
Writes data/reports/<table name>/confound-audit-<split>-<stamp> .md/.json.
The test split is not read unless asked for. Reports hold per-feature
summaries only, no per-video data; keep them under data/ (ignored by Git).
Exits with status 2 on split leakage.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from pulsecheck import __version__
from pulsecheck.dataset import SPLITS, LeakageError, read_sidecar
from pulsecheck.dataset.confounds import MEDIA_FEATURES, R2_FLAG, RHO_FLAG, audit
from pulsecheck.dataset.manifest import file_sha256
from pulsecheck.dataset.media import SHORTCUT_FEATURES
from pulsecheck.dataset.table import join_tables
from pulsecheck.features import PULSE_CLASSIFIER_FEATURES
from pulsecheck.models.training import feature_matrix, labels, load_table


def _fmt(value, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_markdown(report: dict) -> str:
    r = report["audit"]
    cond = report["conditions"]
    lines = [
        f"# PulseCheck confound audit ({cond['split']} split)",
        "",
        f"Generated {report['created_at']} by PulseCheck {report['pulsecheck_version']}.",
        "",
        "> Association only: a flagged feature can still carry pulse signal, and an "
        "unflagged one can still be confounded in ways these media columns do not capture.",
        "",
        "## Conditions",
        "",
        *(f"- **{k}**: {json.dumps(v) if isinstance(v, dict | list) else v}"
          for k, v in cond.items()),
        f"- **rows**: {r['rows']} ({r['real']} real, {r['fake']} fake)",
        f"- **regressor**: {json.dumps(r['regressor'])}",
        "",
        "## (a) Shortcut columns alone",
        "",
        "ROC-AUC with the column itself as the score (higher = fake). Separation = "
        "max(AUC, 1 - AUC); 0.5 is chance.",
        "",
        "| Column | Videos | ROC-AUC | Separation |",
        "|---|---|---|---|",
    ]
    for name, e in r["shortcut_auc"].items():
        lines.append(f"| {name} | {e['n']} | {_fmt(e['roc_auc'])} | {_fmt(e['separation'])} |")
    media = r["media_features"]
    lines += [
        "",
        "## (b) Pulse classifier features vs media",
        "",
        f"Flagged if |Spearman rho| >= {r['thresholds']['abs_spearman']} with any media column "
        f"or cross-validated R^2 from all media columns >= {r['thresholds']['cv_r2']}. "
        f"{len(r['flagged'])} of {len(r['pulse_features'])} features flagged.",
        "",
        f"| Feature | Videos | {' | '.join(f'rho {m}' for m in media)} | CV R^2 | Flag |",
        f"|---|---|{'---|' * len(media)}---|---|",
    ]
    for name, e in r["pulse_features"].items():
        rhos = " | ".join(_fmt(e["spearman"][m], 2) for m in media)
        flag = "; ".join(e["flag_reasons"]) if e["flagged"] else ""
        lines.append(f"| {name} | {e['n']} | {rhos} | {_fmt(e['cv_r2'], 2)} | {flag} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("features", type=Path, help="feature table from extract_features.py")
    parser.add_argument("--media", type=Path, required=True, help="table from probe_media.py")
    parser.add_argument("--manifest", type=Path, help="manifest to join (checks labels/splits)")
    parser.add_argument("--split", default="train", choices=SPLITS)
    parser.add_argument("--out-dir", type=Path, help="default: data/reports/<table name>")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        data = load_table(args.features, join_tables(args.manifest, [args.media]))
    except LeakageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    rows = data.rows[args.split]
    y = labels(rows)
    if len(np.unique(y)) < 2:
        print(f"error: the {args.split} split needs both real and fake videos", file=sys.stderr)
        return 1
    names = list(dict.fromkeys([*PULSE_CLASSIFIER_FEATURES, *SHORTCUT_FEATURES]))
    x = feature_matrix(rows, names)
    columns = {name: x[:, j] for j, name in enumerate(names)}
    unprobed = int(np.sum(~np.isfinite(columns["bitrate_kbps"])))
    result = audit(
        columns,
        y,
        [r["group_id"] or r["video_id"] for r in rows],
        PULSE_CLASSIFIER_FEATURES,
        folds=args.folds,
        seed=args.seed,
        trees=args.trees,
    )

    meta = read_sidecar(args.features)
    dataset = meta.get("dataset") or {}
    now = datetime.now(UTC)
    report = {
        "report_type": "pulsecheck-confound-audit",
        "created_at": now.isoformat(timespec="seconds"),
        "pulsecheck_version": __version__,
        "conditions": {
            "split": args.split,
            "dataset": dataset.get("dataset"),
            "compression": dataset.get("compression"),
            "license": dataset.get("license_notes"),
            "feature_table": args.features.name,
            "feature_table_sha256": file_sha256(args.features),
            "media_table": args.media.name,
            "media_table_sha256": file_sha256(args.media),
            "manifest_sha256": meta.get("manifest_sha256"),
            "failed_rows_excluded": len(data.failed[args.split]),
            "rows_without_media": unprobed,
            "media_features": list(MEDIA_FEATURES),
            "flag_rule": f"|rho| >= {RHO_FLAG} or CV R^2 >= {R2_FLAG}",
        },
        "audit": result,
    }
    out_dir = args.out_dir or Path("data/reports") / args.features.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"confound-audit-{args.split}-{now.strftime('%Y%m%dT%H%M%SZ')}"
    json_path, md_path = out_dir / f"{stem}.json", out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"report: {md_path} and {json_path.name}")
    for name, e in result["shortcut_auc"].items():
        print(f"  {name}: ROC-AUC {_fmt(e['roc_auc'])} (separation {_fmt(e['separation'])})")
    print(f"  flagged pulse features ({len(result['flagged'])}): {', '.join(result['flagged'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
