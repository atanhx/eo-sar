"""Format raw experiment logs into a consistent structured format.

Reads raw training and evaluation log files from the outputs/ or evals/
directories and produces clean, consistently formatted summaries.

Supported log types:
    - Training logs:  epoch-by-epoch loss/metrics + final summary
    - Eval logs:      threshold sweep + metrics + confusion matrix

Usage:
    # Format a single log file:
    python scripts/format_logs.py path/to/logfile.log

    # Format all logs for an experiment:
    python scripts/format_logs.py --exp 3

    # Format all logs across all experiments:
    python scripts/format_logs.py --all

    # Write to file instead of stdout:
    python scripts/format_logs.py --exp 3 --output outputs/exp3/logs/summary.md
"""

import argparse
import re
import sys
from pathlib import Path

RE_EPOCH_LINE = re.compile(r"Epoch\s+(\d+)/(\d+)\s+LR:\s+([\d.e-]+)")
RE_TRAIN_LOSS = re.compile(r"Train Loss:\s+([\d.]+)")
RE_VAL_METRICS = re.compile(
    r"Val IoU:\s+([\d.]+)\s+F1:\s+([\d.]+)\s+Precision:\s+([\d.]+)\s+Recall:\s+([\d.]+)"
)
RE_EPOCH_TIME = re.compile(r"Epoch time:\s+([\d.]+)s\s+Val time:\s+([\d.]+)s")
RE_CHECKPOINT = re.compile(r"Checkpoint saved to .+\(F1:\s+([\d.]+)\)")

RE_THRESH_ROW = re.compile(r"\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")
RE_BEST_THRESH = re.compile(r"Best threshold:\s+([\d.]+)\s+\(Val F1:\s+([\d.]+)\)")
RE_ORACLE_BEST = re.compile(r"Oracle best:\s+threshold=([\d.]+)\s+F1=([\d.]+)")

RE_RESULT_SPLIT = re.compile(r"RESULTS:\s+(\w+)\s+SPLIT")
RE_IOU = re.compile(r"IoU:\s+([\d.]+)")
RE_PREC = re.compile(r"Precision:\s+([\d.]+)")
RE_RECALL = re.compile(r"Recall:\s+([\d.]+)")
RE_F1 = re.compile(r"F1 Score:\s+([\d.]+)")
RE_CM_NC = re.compile(r"Actual No-Change\s+(\d+)\s+(\d+)")
RE_CM_C = re.compile(r"Actual Change\s+(\d+)\s+(\d+)")
RE_MISS = re.compile(r"Miss rate.*?:\s+([\d.]+%)")
RE_CAUGHT = re.compile(r"model caught\s+([\d.]+%)")
RE_FALSE_ALARM = re.compile(r"False alarm rate.*?:\s+([\d.]+%)")

RE_BEST_F1_SUMMARY = re.compile(r"Best validation F1:\s+([\d.]+)\s+at epoch\s+(\d+)")
RE_TOTAL_TIME = re.compile(r"Total training time:\s+([\d.]+)\s+hours")
RE_AVG_EPOCH = re.compile(r"Avg time per epoch:\s+([\d.]+)s")
RE_DEVICE = re.compile(r"Device:\s+(\S+)")
RE_ENCODER = re.compile(r"Encoder:\s+(\S+)")
RE_PARAMS = re.compile(r"Parameters:\s+([\d,]+)")

RE_LOADED_EPOCH = re.compile(r"Loaded checkpoint from epoch\s+(\d+)")
RE_SAMPLES_INFO = re.compile(
    r"\[(\w+)\] Found (\d+) samples:\s+(\d+) with change.*?(\d+) (?:no-change|purely)"
)


def classify_log(text: str) -> str:
    """Classify log content as 'train', 'eval', or 'unknown'."""
    if "TRAINING COMPLETE" in text or "Starting training" in text:
        return "train"
    if "Threshold sweep" in text or "RESULTS:" in text:
        return "eval"
    return "unknown"


def format_train_log(text: str, filename: str) -> str:
    """Format a training log into structured output."""
    lines = []
    lines.append(f"# Training Log: {filename}")
    lines.append("")

    # Device and encoder
    m = RE_DEVICE.search(text)
    if m:
        lines.append(f"**Device:** {m.group(1)}")
    m = RE_ENCODER.search(text)
    if m:
        lines.append(f"**Encoder:** {m.group(1)}")
    m = RE_PARAMS.search(text)
    if m:
        lines.append(f"**Parameters:** {m.group(1)}")
    lines.append("")

    # Summary
    m_best = RE_BEST_F1_SUMMARY.search(text)
    m_time = RE_TOTAL_TIME.search(text)
    m_avg = RE_AVG_EPOCH.search(text)
    if m_best:
        lines.append(f"**Best Val F1:** {m_best.group(1)} (epoch {m_best.group(2)})")
    if m_time:
        lines.append(f"**Total Time:** {m_time.group(1)} hours")
    if m_avg:
        lines.append(f"**Avg Epoch Time:** {m_avg.group(1)}s")
    lines.append("")

    # Epoch table
    epochs = []
    current_epoch = {}
    for line in text.splitlines():
        m = RE_EPOCH_LINE.match(line.strip())
        if m:
            if current_epoch:
                epochs.append(current_epoch)
            current_epoch = {
                "epoch": int(m.group(1)),
                "total": int(m.group(2)),
                "lr": m.group(3),
            }
            continue

        m = RE_TRAIN_LOSS.search(line)
        if m and current_epoch:
            current_epoch["loss"] = float(m.group(1))

        m = RE_VAL_METRICS.search(line)
        if m and current_epoch:
            current_epoch["iou"] = float(m.group(1))
            current_epoch["f1"] = float(m.group(2))
            current_epoch["prec"] = float(m.group(3))
            current_epoch["recall"] = float(m.group(4))

        m = RE_EPOCH_TIME.search(line)
        if m and current_epoch:
            current_epoch["time"] = float(m.group(1))

        m = RE_CHECKPOINT.search(line)
        if m and current_epoch:
            current_epoch["ckpt"] = True

    if current_epoch:
        epochs.append(current_epoch)

    if epochs:
        lines.append("## Epoch Summary")
        lines.append("")
        lines.append(
            "| Epoch | Loss   | Val F1  | Val IoU | Precision | Recall | Time (s) | Ckpt |"
        )
        lines.append(
            "|------:|-------:|--------:|--------:|----------:|-------:|---------:|:----:|"
        )
        for e in epochs:
            ckpt = "✓" if e.get("ckpt") else ""
            lines.append(
                f"| {e.get('epoch', '?'):>5} "
                f"| {e.get('loss', 0):.4f} "
                f"| {e.get('f1', 0):.4f} "
                f"| {e.get('iou', 0):.4f} "
                f"| {e.get('prec', 0):.4f}    "
                f"| {e.get('recall', 0):.4f} "
                f"| {e.get('time', 0):>8.1f} "
                f"| {ckpt:^4} |"
            )
        lines.append("")

    return "\n".join(lines)


def format_eval_log(text: str, filename: str) -> str:
    """Format an evaluation log into structured output."""
    lines = []
    lines.append(f"# Evaluation Log: {filename}")
    lines.append("")

    # Loaded checkpoint
    m = RE_LOADED_EPOCH.search(text)
    if m:
        lines.append(f"**Checkpoint Epoch:** {m.group(1)}")

    m = RE_DEVICE.search(text)
    if m:
        lines.append(f"**Device:** {m.group(1)}")

    # Samples info
    for m in RE_SAMPLES_INFO.finditer(text):
        lines.append(
            f"**[{m.group(1)}] Samples:** {m.group(2)} total "
            f"({m.group(3)} with change, {m.group(4)} no-change)"
        )
    lines.append("")

    # Threshold sweep
    thresh_rows = []
    for m in RE_THRESH_ROW.finditer(text):
        thresh_rows.append(
            {
                "threshold": float(m.group(1)),
                "f1": float(m.group(2)),
                "precision": float(m.group(3)),
                "recall": float(m.group(4)),
                "iou": float(m.group(5)),
            }
        )

    if thresh_rows:
        lines.append("## Threshold Sweep")
        lines.append("")
        lines.append("| Threshold |     F1 | Precision |  Recall |    IoU |")
        lines.append("|----------:|-------:|----------:|--------:|-------:|")
        for r in thresh_rows:
            lines.append(
                f"| {r['threshold']:>9.2f} "
                f"| {r['f1']:.4f} "
                f"| {r['precision']:.4f}    "
                f"| {r['recall']:.4f} "
                f"| {r['iou']:.4f} |"
            )
        lines.append("")

    # Best threshold
    m = RE_BEST_THRESH.search(text)
    if m:
        lines.append(f"**Best Threshold:** {m.group(1)} (Val F1: {m.group(2)})")
        lines.append("")

    # Results sections
    for result_match in RE_RESULT_SPLIT.finditer(text):
        split_name = result_match.group(1)
        # Find the metrics block after this match
        start_pos = result_match.end()
        block = text[start_pos : start_pos + 800]

        lines.append(f"## Results: {split_name} Split")
        lines.append("")

        m = RE_IOU.search(block)
        iou = m.group(1) if m else "?"
        m = RE_PREC.search(block)
        prec = m.group(1) if m else "?"
        m = RE_RECALL.search(block)
        recall = m.group(1) if m else "?"
        m = RE_F1.search(block)
        f1 = m.group(1) if m else "?"

        lines.append("| Metric    | Value  |")
        lines.append("|-----------|--------|")
        lines.append(f"| IoU       | {iou} |")
        lines.append(f"| Precision | {prec} |")
        lines.append(f"| Recall    | {recall} |")
        lines.append(f"| F1 Score  | {f1} |")
        lines.append("")

        # Confusion matrix
        m_nc = RE_CM_NC.search(block)
        m_c = RE_CM_C.search(block)
        if m_nc and m_c:
            lines.append("### Confusion Matrix")
            lines.append("")
            lines.append("|                  | Pred No-Change | Pred Change |")
            lines.append("|:-----------------|---------------:|------------:|")
            lines.append(
                f"| Actual No-Change | {int(m_nc.group(1)):>14,} | {int(m_nc.group(2)):>11,} |"
            )
            lines.append(
                f"| Actual Change    | {int(m_c.group(1)):>14,} | {int(m_c.group(2)):>11,} |"
            )
            lines.append("")

        # Error profile
        m_miss = RE_MISS.search(block)
        m_caught = RE_CAUGHT.search(block)
        m_fa = RE_FALSE_ALARM.search(block)
        if m_miss:
            lines.append(f"- **Miss rate:** {m_miss.group(1)}")
        if m_caught:
            lines.append(f"- **Caught:** {m_caught.group(1)}")
        if m_fa:
            lines.append(f"- **False alarm rate:** {m_fa.group(1)}")
        lines.append("")

    # Oracle threshold
    m = RE_ORACLE_BEST.search(text)
    if m:
        lines.append("## Oracle Threshold (Diagnostic)")
        lines.append("")
        lines.append(f"**Oracle Best:** threshold={m.group(1)}, F1={m.group(2)}")
        lines.append("")

    return "\n".join(lines)


def format_log(filepath: Path) -> str:
    """Read and format a single log file."""
    text = filepath.read_text(encoding="utf-8", errors="replace")
    log_type = classify_log(text)

    if log_type == "train":
        return format_train_log(text, filepath.name)
    elif log_type == "eval":
        return format_eval_log(text, filepath.name)
    else:
        # Return raw with a header
        return f"# Log: {filepath.name}\n\n```\n{text}\n```\n"


def find_logs_for_experiment(exp_num: int) -> list[Path]:
    """Find all log files for a given experiment number."""
    project_root = Path(__file__).resolve().parent.parent
    log_files = []

    # New structure: outputs/expN/logs/
    new_dir = project_root / "outputs" / f"exp{exp_num}" / "logs"
    if new_dir.exists():
        log_files.extend(sorted(new_dir.glob("*.log")))
        log_files.extend(sorted(new_dir.glob("*.txt")))

    # Legacy structure: evals/expN/logs_*/
    legacy_dir = project_root / "evals" / f"exp{exp_num}"
    if legacy_dir.exists():
        for sub in sorted(legacy_dir.iterdir()):
            if sub.is_dir() and sub.name.startswith("logs"):
                log_files.extend(sorted(sub.glob("*.log")))
                log_files.extend(sorted(sub.glob("*.txt")))
        # Also check top-level txt/md files in the exp dir
        log_files.extend(sorted(legacy_dir.glob("*.txt")))
        log_files.extend(sorted(legacy_dir.glob("*.md")))

    return log_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Format raw experiment logs into consistent structured output.",
        epilog="Outputs markdown-formatted summaries of training and evaluation logs.",
    )
    parser.add_argument(
        "logfile",
        nargs="?",
        type=str,
        default=None,
        help="Path to a single log file to format.",
    )
    parser.add_argument(
        "--exp",
        type=int,
        default=None,
        help="Experiment number (e.g. 3). Formats all logs for that experiment.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="format_all",
        help="Format logs for all experiments.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Write output to file instead of stdout.",
    )
    args = parser.parse_args()

    if not args.logfile and args.exp is None and not args.format_all:
        parser.print_help()
        sys.exit(1)

    output_parts = []

    if args.logfile:
        p = Path(args.logfile)
        if not p.exists():
            print(f"Error: file not found: {p}", file=sys.stderr)
            sys.exit(1)
        output_parts.append(format_log(p))

    elif args.exp is not None:
        logs = find_logs_for_experiment(args.exp)
        if not logs:
            print(f"No logs found for experiment {args.exp}", file=sys.stderr)
            sys.exit(1)
        header = f"# Experiment {args.exp} — All Logs\n\n"
        output_parts.append(header)
        for lf in logs:
            output_parts.append(format_log(lf))
            output_parts.append("\n---\n")

    elif args.format_all:
        for exp_num in range(1, 10):
            logs = find_logs_for_experiment(exp_num)
            if logs:
                output_parts.append(f"# Experiment {exp_num}\n\n")
                for lf in logs:
                    output_parts.append(format_log(lf))
                    output_parts.append("\n---\n")
                output_parts.append("\n")

    result = "\n".join(output_parts)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(result, encoding="utf-8")
        print(f"Formatted output written to {out_path}")
    else:
        print(result)


if __name__ == "__main__":
    main()
