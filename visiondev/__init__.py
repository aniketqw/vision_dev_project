"""VisionDev: find out why an image classifier fails.

    from visiondev import DebugLogger
    trainer = pl.Trainer(callbacks=[DebugLogger(diagnose=True)])   # train as usual

The callback tags every misclassified image with a distortion tagger, groups the images by tag,
picks typical/outlier ones (ResNet18 -> PCA -> t-SNE), asks a vision-language model about them and writes report.md.
"""
from .callback import DebugLogger
from .tagger.infer import Tagger

__all__ = ["DebugLogger", "Tagger", "diagnose", "DiagnosisConfig"]
__version__ = "0.2.0"


def __getattr__(name):  # keep `import visiondev` light: the diagnosis stack loads on first use
    if name in ("diagnose", "DiagnosisConfig"):
        from . import pipeline
        return getattr(pipeline, name)
    raise AttributeError(name)
