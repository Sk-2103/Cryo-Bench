#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


SECTION_RE = re.compile(r"\[(?P<split>[^\]]+)\] ------- (?P<metric>.+?) --------")
CLASS_RE = re.compile(r"^(?P<class>.+?)\s+(?P<value>-?\d+\.\d+)\s*$")
MEAN_RE = re.compile(r"^\[[^\]]+\]\s*Mean\s+(?P<value>-?\d+\.\d+)\s*$")
ACC_RE = re.compile(r"Mean Accuracy:\s*(?P<value>-?\d+\.\d+)")


def metric_line(raw_line: str) -> str:
    line = raw_line.strip()
    bracket = line.find("[")
    if bracket >= 0:
        return line[bracket:]
    acc = line.find("Mean Accuracy:")
    if acc >= 0:
        return line[acc:]
    return line


def latest_metric_block(text: str, split: str = "test") -> dict[str, float]:
    blocks = []
    current = None
    for raw_line in text.splitlines():
        line = metric_line(raw_line)
        match = SECTION_RE.match(line)
        if match:
            current = {"split": match.group("split"), "metric": match.group("metric").strip()}
            blocks.append(current)
            continue
        acc_match = ACC_RE.search(line)
        if acc_match and current is not None and current.get("split") == split:
            blocks.append({"split": split, "metric": "Accuracy", "Mean": float(acc_match.group("value"))})
            continue
        if current is None or current["split"] != split:
            continue
        mean_match = MEAN_RE.match(line)
        if mean_match:
            current["Mean"] = float(mean_match.group("value"))
            continue
        class_match = CLASS_RE.match(line)
        if class_match:
            current[class_match.group("class").strip()] = float(class_match.group("value"))

    metrics = {}
    for block in blocks:
        if block.get("split") != split:
            continue
        metric = block.get("metric")
        if metric == "IoU":
            metrics["mIoU"] = block.get("Mean")
            metrics["IoU_background"] = block.get("Background")
            metrics["IoU_glacial_lake"] = block.get("Glacial Lake")
        elif metric == "F1-score":
            metrics["mF1"] = block.get("Mean")
            metrics["F1_background"] = block.get("Background")
            metrics["F1_glacial_lake"] = block.get("Glacial Lake")
        elif metric == "Precision":
            metrics["mPrecision"] = block.get("Mean")
            metrics["Precision_glacial_lake"] = block.get("Glacial Lake")
        elif metric == "Recall":
            metrics["mRecall"] = block.get("Mean")
            metrics["Recall_glacial_lake"] = block.get("Glacial Lake")
        elif metric == "Accuracy":
            metrics["mAcc"] = block.get("Mean")
    return metrics


def find_runs(runs_dir: Path) -> list[Path]:
    if not runs_dir.exists():
        return []
    return sorted(p for p in runs_dir.iterdir() if p.is_dir())


def find_log(run: Path) -> Path | None:
    candidates = [
        run / "test.log",
        run / "test.log-0",
        run / "train.log",
        run / "train.log-0",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    logs = sorted(run.glob("*.log*"))
    return logs[-1] if logs else None


def encoder_from_run(run: Path) -> str:
    config = run / "configs" / "config.yaml"
    if config.exists():
        for line in config.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("_target_: pangaea.encoders."):
                return line.rsplit(".", 1)[-1]
    parts = run.name.split("_")
    if len(parts) >= 4:
        return parts[3]
    return run.name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="/media/turtle-ssd/users/skaushik/GLB_data_results/pangaea")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    rows = []
    for run in find_runs(results_dir / "runs"):
        log_path = find_log(run)
        if log_path is None or not log_path.exists():
            continue
        status = "complete"
        metrics = latest_metric_block(log_path.read_text(encoding="utf-8", errors="replace")) if log_path.exists() else {}
        if log_path.exists() and not metrics.get("mIoU"):
            status = "no_test_metrics"
        rows.append(
            {
                "run": run.name,
                "encoder": encoder_from_run(run),
                "status": status,
                **metrics,
            }
        )

    fieldnames = [
        "encoder",
        "status",
        "mIoU",
        "IoU_glacial_lake",
        "mF1",
        "F1_glacial_lake",
        "mPrecision",
        "Precision_glacial_lake",
        "mRecall",
        "Recall_glacial_lake",
        "mAcc",
        "run",
    ]
    csv_path = results_dir / "results.csv"
    results_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    md_path = results_dir / "results_table.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# GLB Random PANGAEA Results\n\n")
        f.write("| Encoder | Status | mIoU | Lake IoU | mF1 | Lake F1 | mAcc | Run |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            fmt = lambda key: "" if row.get(key) is None else f"{row[key]:.3f}"
            f.write(
                f"| {row['encoder']} | {row['status']} | {fmt('mIoU')} | "
                f"{fmt('IoU_glacial_lake')} | {fmt('mF1')} | "
                f"{fmt('F1_glacial_lake')} | {fmt('mAcc')} | {row['run']} |\n"
            )

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
