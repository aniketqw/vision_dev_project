"""Command line: re-run the diagnosis on an existing run folder.

    visiondev diagnose visiondev_runs/20260101_120000 --port 8000 --model Qwen/Qwen2.5-VL-7B-Instruct
    visiondev diagnose <run_dir> --no-vlm        # statistics-only report
"""
import argparse

from .config import DEFAULT_MODEL, DEFAULT_PORT


def main(argv=None):
    ap = argparse.ArgumentParser(prog="visiondev")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("diagnose", help="select failures, ask the VLM, write report.md for a run folder")
    d.add_argument("run_dir")
    d.add_argument("--no-vlm", action="store_true", help="statistics-only report")
    d.add_argument("--port", type=int, default=DEFAULT_PORT)
    d.add_argument("--model", default=DEFAULT_MODEL)
    d.add_argument("--samples", type=int, default=6, help="images per bucket (3 typical + 3 outlier)")
    d.add_argument("--rag", action="store_true", help="use cross-run failure memory")
    a = ap.parse_args(argv)
    from .pipeline import DiagnosisConfig, diagnose
    print("report:", diagnose(a.run_dir, DiagnosisConfig(use_vlm=not a.no_vlm, vlm_port=a.port, vlm_model=a.model,
                                                         samples=a.samples, rag=a.rag)))


if __name__ == "__main__":
    main()
