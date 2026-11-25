"""After training: select representative failures, ask a vision-language model, write the report.

    from visiondev.pipeline import diagnose, DiagnosisConfig
    report = diagnose("visiondev_runs/20260101_120000", DiagnosisConfig(vlm_port=8000))

The run folder is the one `DebugLogger` wrote. It must contain training_log_*.json and misclassified_*.json.
Output (inside the same folder):
    misclassified_<ts>_images/<tag>/<hash>.png   failures grouped by the tag the tagger gave them
    selection/distortion_report.json             3 typical + 3 outlier picked per tag
    selection/tsne.png                           cluster map
    report.md                                    the diagnostic report (the deliverable)
If no VLM server answers, the report is still written, with statistics only and no model analysis.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from .config import ANALYSIS_TYPES, DEFAULT_MODEL, DEFAULT_PORT, DEFAULT_RAG_INDEX
from .log import logger
from .selection import archetypes


@dataclass
class DiagnosisConfig:
    use_vlm: bool = True            # False = statistics-only report
    vlm_model: str = DEFAULT_MODEL  # model name sent to the OpenAI-compatible endpoint
    vlm_port: int = DEFAULT_PORT    # endpoint is http://localhost:<port>/v1
    server_timeout: int = 60        # seconds to wait for the endpoint before giving up on the VLM
    samples: int = 6                # images per bucket: 3 typical then 3 outlier
    seed: int = 42
    max_per_category: int = 300     # cap per tag before clustering
    rag: bool = False               # cross-run failure memory (needs several runs in the parent folder)
    tool_trace: bool = True         # include the verification tool results in the report


def _latest(folder: Path, pattern: str) -> Path:
    files = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"no {pattern} in {folder}")
    return files[-1]


def diagnose(run_dir: Union[str, Path], config: Optional[DiagnosisConfig] = None) -> Path:
    """Run selection and diagnosis for one training run and return the path of report.md."""
    cfg = config or DiagnosisConfig()
    run_dir = Path(run_dir).resolve()
    training_log, mc_path = _latest(run_dir, "training_log_*.json"), _latest(run_dir, "misclassified_*.json")

    # 1. group images by tag, ResNet18 -> PCA -> t-SNE, pick typical + outlier
    report_json, plot_png = run_dir / "selection" / "distortion_report.json", run_dir / "selection" / "tsne.png"
    archetypes.run(mc_path, report_json, plot_png, cfg.max_per_category)

    # 2. heavy imports only now, so `import visiondev` stays light
    from .diagnosis.chains import build_llm
    from .diagnosis.data_loader import (load_all_misclassified, load_distortion_report,
                                        load_misclassified_stats, load_training_summary)
    from .diagnosis.graph import AnalysisState, build_graph
    from .diagnosis.image_sampler import gather_images_for_distortion, semantic_deduplicate
    from .diagnosis.rag import RAGStore
    from .diagnosis.renderer import _fallback_recommendations, generate_dynamic_recommendations, render_report
    from .diagnosis.server_utils import wait_for_server

    training = load_training_summary(training_log)
    mc_stats = load_misclassified_stats(mc_path)
    dist_report = load_distortion_report(report_json)
    images_dir = mc_path.parent / f"{mc_path.stem}_images"

    rag_store = None
    if cfg.rag:
        try:
            rag_store = RAGStore.load_or_build(index_path=run_dir / "rag_index.npz",
                                               all_mc_stats=load_all_misclassified(run_dir.parent), force_rebuild=False)
        except Exception as exc:  # RAG is optional; never block the report
            logger.warning(f"RAG unavailable: {exc}")

    llm = graph = None
    if cfg.use_vlm:
        if wait_for_server(cfg.vlm_port, timeout=cfg.server_timeout, poll_interval=5):
            llm = build_llm(cfg.vlm_model, cfg.vlm_port)
            graph = build_graph(llm, rag_store, mc_stats)
        else:
            logger.warning("No VLM server answered; writing a statistics-only report.")

    states = {}
    for tag in ANALYSIS_TYPES:
        items = gather_images_for_distortion(tag, dist_report, mc_stats, images_dir, cfg.samples, cfg.seed)
        items = semantic_deduplicate(items)
        state = AnalysisState(dist_type=tag, items=items)
        states[tag] = graph.run(state) if (graph is not None and items) else state

    root_causes = {t: (s.structured_output.root_cause if s.structured_output else (s.hypothesis_text or "")[:200])
                   for t, s in states.items() if s.structured_output or s.hypothesis_text}
    recs = (generate_dynamic_recommendations(llm=llm, mc_stats=mc_stats, root_causes=root_causes,
                                             model_name=cfg.vlm_model, port=cfg.vlm_port)
            if llm is not None else _fallback_recommendations(mc_stats))

    out = run_dir / "report.md"
    render_report(training=training, mc_stats=mc_stats, per_type_states=states, recommendations=recs,
                  output_path=out, tool_trace=cfg.tool_trace)
    logger.info(f"Report written to {out}")
    return out
