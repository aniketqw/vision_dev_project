"""Cluster the misclassified images and pick 3 typical + 3 outlier per tag.

    python scripts/02_select_archetypes.py            # uses the newest outputs/logs/misclassified_*.json
Writes outputs/reports/distortion_report.json and outputs/reports/distortion_tsne.png
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from visiondev.config import DEFAULT_LOGS_DIR, DEFAULT_REPORTS_DIR
from visiondev.selection import archetypes

ap = argparse.ArgumentParser()
ap.add_argument("--misclassified", type=Path, default=None)
ap.add_argument("--report", type=Path, default=DEFAULT_REPORTS_DIR / "distortion_report.json")
ap.add_argument("--plot", type=Path, default=DEFAULT_REPORTS_DIR / "distortion_tsne.png")
ap.add_argument("--max-per-category", type=int, default=300)
a = ap.parse_args()

mc = a.misclassified or max(DEFAULT_LOGS_DIR.glob("misclassified_*.json"), key=lambda p: p.stat().st_mtime)
rep = archetypes.run(mc, a.report, a.plot, a.max_per_category)
print("counts:", rep["counts"], "\nreport:", a.report, "\nplot:", a.plot)
