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


def latest_metric_block(text: str, target_class: str, split: str = "test") -> dict[str, float]:
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
            metrics["IoU_target"] = block.get(target_class)
        elif metric == "F1-score":
            metrics["mF1"] = block.get("Mean")
            metrics["F1_target"] = block.get(target_class)
        elif metric == "Precision":
            metrics["mPrecision"] = block.get("Mean")
            metrics["Precision_target"] = block.get(target_class)
        elif metric == "Recall":
            metrics["mRecall"] = block.get("Mean")
            metrics["Recall_target"] = block.get(target_class)
        elif metric == "Accuracy":
            metrics["mAcc"] = block.get("Mean")
    return metrics


def find_log(run: Path) -> Path | None:
    for candidate in [run / "test.log", run / "test.log-0", run / "train.log", run / "train.log-0"]:
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
    return run.name


def fmt(row: dict, key: str) -> str:
    return "" if row.get(key) is None else f"{row[key]:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default="/media/turtle-ssd/users/skaushik/Cryo-Data/Benchmark/Shelf-Bench/output/pangaea",
    )
    parser.add_argument("--target-class", default="Ice Shelf")
    parser.add_argument("--title", default="Shelf-Bench PANGAEA Results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    rows = []
    for run in sorted((results_dir / "runs").iterdir()) if (results_dir / "runs").exists() else []:
        if not run.is_dir():
            continue
        log_path = find_log(run)
        if log_path is None:
            continue
        metrics = latest_metric_block(
            log_path.read_text(encoding="utf-8", errors="replace"),
            target_class=args.target_class,
        )
        rows.append(
            {
                "run": run.name,
                "encoder": encoder_from_run(run),
                "status": "complete" if metrics.get("mIoU") is not None else "no_test_metrics",
                **metrics,
            }
        )

    fieldnames = [
        "encoder",
        "status",
        "mIoU",
        "IoU_target",
        "mF1",
        "F1_target",
        "mPrecision",
        "Precision_target",
        "mRecall",
        "Recall_target",
        "mAcc",
        "run",
    ]
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    with (results_dir / "results_table.md").open("w", encoding="utf-8") as f:
        f.write(f"# {args.title}\n\n")
        f.write("## Summary\n\n")
        f.write("| Run ID | Encoder | Status | mIoU | Ice Shelf IoU | mF1 | Ice Shelf F1 | mAcc |\n")
        f.write("|:------:|:--------|:-------|-----:|--------------:|----:|-------------:|-----:|\n")
        for idx, row in enumerate(rows, start=1):
            run_id = f"R{idx:02d}"
            f.write(
                f"| {run_id} | {row['encoder']} | {row['status']} | {fmt(row, 'mIoU')} | "
                f"{fmt(row, 'IoU_target')} | {fmt(row, 'mF1')} | "
                f"{fmt(row, 'F1_target')} | {fmt(row, 'mAcc')} |\n"
            )
        f.write("\n## Run Directories\n\n")
        f.write("| Run ID | Run directory |\n")
        f.write("|:------:|:--------------|\n")
        for idx, row in enumerate(rows, start=1):
            run_id = f"R{idx:02d}"
            f.write(f"| {run_id} | `{row['run']}` |\n")

    print(f"Wrote {results_dir / 'results.csv'}")
    print(f"Wrote {results_dir / 'results_table.md'}")


if __name__ == "__main__":
    main()
