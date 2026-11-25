"""Build and validate the fault -> fix RAG knowledge base for the VisionDev SimpleCNN.

Usage (from rag_kb/):  ../pipe/.venv/bin/python build_cards.py [--smoke]
Writes fix_cards.jsonl. --smoke additionally executes snippets that only need torch/PIL.

evidence_level:
  literature  - claim traced to a paper/doc that was opened and checked
  standard    - widely used technique; implementation written here, effect NOT yet measured
  hypothesis  - our own reasoning; must be confirmed by a retrain-and-score run
verified_by_run is filled in later with measured before/after accuracy per distortion
(this is the feedback signal that lets the retriever prefer fixes that actually worked).
"""
import ast
import json
import sys
from pathlib import Path

TARGET = "pipe/test.py SimpleCNN + setup_data_loaders (3x conv+maxpool, no augmentation)"
NORM = "transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))"

CARDS = [
    dict(
        id="blur-gaussian-aug-torchvision", distortions=["blur"],
        symptom="Misclassified images are soft/out of focus; blur cluster dominates the report.",
        cause="Training set has only sharp CIFAR-10 images, so edge/texture filters are never exposed to blur.",
        fix="Randomly blur a fraction of TRAIN images only; keep the validation transform clean.",
        code='''from torchvision import transforms
train_tf = transforms.Compose([
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.3),
    transforms.ToTensor(),
    %s,
])
# in setup_data_loaders: datasets.CIFAR10(..., train=True, transform=train_tf)''' % NORM,
        expected_effect="Higher accuracy on blurred inputs; small or no change on clean accuracy.",
        evidence_level="standard", source="torchvision.transforms.GaussianBlur (torchvision docs); effect unmeasured here"),
    dict(
        id="blur-albumentations-multi", distortions=["blur"],
        symptom="Blur failures include motion/defocus style blur, not only Gaussian.",
        cause="A single blur type does not cover the blur varieties in the data.",
        fix="Use a OneOf of several blur transforms.",
        code='''import albumentations as A
blur_aug = A.OneOf([A.GaussianBlur(blur_limit=(3, 5)), A.MotionBlur(blur_limit=5), A.Defocus(radius=(1, 3))], p=0.3)
# apply on numpy HWC uint8 images: img = blur_aug(image=img)["image"]''',
        expected_effect="Broader blur robustness than a single Gaussian blur.",
        evidence_level="standard", source="Albumentations documentation (albumentations.ai); transform argument names vary by version, check installed version"),
    dict(
        id="noise-gaussian-aug", distortions=["noise"],
        symptom="Misclassified images show grain/speckle; noise cluster in the report.",
        cause="Model relies on high-frequency pixel patterns that noise destroys; never trained on noisy inputs.",
        fix="Add random Gaussian noise to a fraction of training tensors before Normalize.",
        code='''import torch
from torchvision import transforms

class AddGaussianNoise:
    def __init__(self, std_range=(0.02, 0.15), p=0.3):
        self.std_range, self.p = std_range, p
    def __call__(self, x):  # x: float tensor in [0, 1]
        if torch.rand(1).item() > self.p:
            return x
        std = torch.empty(1).uniform_(*self.std_range).item()
        return (x + torch.randn_like(x) * std).clamp(0, 1)

train_tf = transforms.Compose([transforms.ToTensor(), AddGaussianNoise(), %s])''' % NORM,
        expected_effect="Higher accuracy on noisy inputs; watch clean accuracy for a small drop.",
        evidence_level="standard", source="standard noise augmentation; implementation written here, effect unmeasured"),
    dict(
        id="noise-iso-albumentations", distortions=["noise"],
        symptom="Noise looks sensor-like (colour grain) rather than uniform Gaussian.",
        cause="Camera-style noise differs statistically from additive Gaussian noise.",
        fix="Use ISONoise (optionally with GaussNoise) in the train pipeline.",
        code='''import albumentations as A
noise_aug = A.OneOf([A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)), A.GaussNoise()], p=0.3)''',
        expected_effect="Robustness to sensor-style noise.",
        evidence_level="standard", source="Albumentations docs (explore.albumentations.ai, ISONoise); parameter names vary by version"),
    dict(
        id="jpeg-pil-roundtrip-aug", distortions=["jpeg"],
        symptom="Misclassified images show blocking/ringing artifacts; jpeg cluster in the report.",
        cause="Model never saw compression artifacts; 8x8 block edges shift its low-level features.",
        fix="Re-encode a fraction of training images as JPEG at a random quality before ToTensor.",
        code='''import io, random
from PIL import Image
from torchvision import transforms

class RandomJPEG:
    def __init__(self, quality_range=(20, 80), p=0.3):
        self.quality_range, self.p = quality_range, p
    def __call__(self, img):
        if random.random() > self.p:
            return img
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(*self.quality_range))
        buf.seek(0)
        return Image.open(buf).convert("RGB")

train_tf = transforms.Compose([RandomJPEG(), transforms.ToTensor(), %s])''' % NORM,
        expected_effect="Higher accuracy on compressed inputs.",
        evidence_level="standard", source="standard compression augmentation; implementation written here, effect unmeasured"),
    dict(
        id="jpeg-albumentations-quality", distortions=["jpeg"],
        symptom="Compression artifacts of varying strength.",
        cause="Quality level varies; one fixed setting under-covers it.",
        fix="ImageCompression with a quality range.",
        code='''import albumentations as A
jpeg_aug = A.ImageCompression(quality_range=(30, 90), p=0.3)  # default range is (99, 100), i.e. almost no effect''',
        expected_effect="Robustness across compression strengths.",
        evidence_level="literature", source="https://albumentations.ai/docs/api-reference/albumentations/augmentations/pixel/compression/ (quality_range argument; older versions used quality_lower/quality_upper)"),
    dict(
        id="pixelate-downscale-aug", distortions=["pixelate"],
        symptom="Misclassified images look blocky/low-resolution; pixelate cluster in the report.",
        cause="Detail is lost to downscale-then-upscale; the model has not seen blocky edges.",
        fix="Randomly downscale then upscale training images with nearest-neighbour resampling.",
        code='''import random
from PIL import Image
from torchvision import transforms

class RandomPixelate:
    def __init__(self, scale_range=(0.4, 0.8), p=0.3):
        self.scale_range, self.p = scale_range, p
    def __call__(self, img):
        if random.random() > self.p:
            return img
        w, h = img.size
        s = random.uniform(*self.scale_range)
        small = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.NEAREST)
        return small.resize((w, h), Image.NEAREST)

train_tf = transforms.Compose([RandomPixelate(), transforms.ToTensor(), %s])''' % NORM,
        expected_effect="Higher accuracy on pixelated inputs.",
        evidence_level="standard", source="same idea as Albumentations Downscale (scale_range; albumentations.ai docs); implementation written here"),
    dict(
        id="augmix-torchvision", distortions=["blur", "noise", "jpeg", "pixelate", "mixed"],
        symptom="Several distortion clusters at once, or outliers with mixed/ambiguous distortions.",
        cause="Model is brittle to many unseen corruptions; per-distortion augmentation does not scale.",
        fix="Train with AugMix: mix several random augmentation chains with the original image.",
        code='''from torchvision import transforms
train_tf = transforms.Compose([
    transforms.AugMix(severity=3, mixture_width=3),   # operates on PIL/uint8 images, before ToTensor
    transforms.ToTensor(),
    %s,
])  # requires a torchvision version that ships transforms.AugMix''' % NORM,
        expected_effect="Paper reports roughly halving CIFAR-10 corruption error (28.4% -> 12.4%) with the full method incl. consistency loss.",
        evidence_level="literature", source="https://arxiv.org/abs/1912.02781 (AugMix, Hendrycks et al., ICLR 2020); numbers are for the paper's models, not SimpleCNN"),
    dict(
        id="augmix-jsd-consistency", distortions=["blur", "noise", "jpeg", "pixelate", "mixed"],
        symptom="AugMix helps only partly; predictions flip between clean and corrupted versions of one image.",
        cause="Plain cross-entropy does not force consistent predictions across augmented views.",
        fix="Add the Jensen-Shannon consistency loss over clean and two augmented views.",
        code='''import torch, torch.nn.functional as F

def jsd_consistency(logits_clean, logits_a, logits_b):
    p = [F.softmax(l, dim=1) for l in (logits_clean, logits_a, logits_b)]
    m = torch.clamp(sum(p) / 3, 1e-7, 1).log()
    return sum(F.kl_div(m, pi, reduction="batchmean") for pi in p) / 3
# loss = CE(logits_clean, y) + 12 * jsd_consistency(logits_clean, logits_a, logits_b)''',
        expected_effect="Larger corruption-robustness gain than AugMix augmentation alone (per the paper's ablation).",
        evidence_level="literature", source="https://arxiv.org/abs/1912.02781 (AugMix, JSD consistency loss); weight 12 is the paper's CIFAR value, tune for SimpleCNN"),
    dict(
        id="blurpool-antialias-simplecnn", distortions=["pixelate", "noise", "blur"],
        symptom="Predictions change under small shifts or sampling changes; pixelate/aliasing-like failures.",
        cause="Max-pool with stride 2 aliases high-frequency content; the model becomes unstable to such inputs.",
        fix="Replace each MaxPool2d(2,2) with dense max-pool (stride 1) followed by a fixed blur + stride-2 subsample.",
        code='''import torch, torch.nn as nn, torch.nn.functional as F

class BlurPool2d(nn.Module):
    def __init__(self, channels, stride=2):
        super().__init__()
        k = torch.tensor([1., 2., 1.]); k = k[:, None] * k[None, :]; k = k / k.sum()
        self.register_buffer("kernel", k[None, None].repeat(channels, 1, 1, 1))
        self.stride, self.channels = stride, channels
    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1), mode="reflect")
        return F.conv2d(x, self.kernel, stride=self.stride, groups=self.channels)

# SimpleCNN.__init__: self.bp1, self.bp2, self.bp3 = BlurPool2d(32), BlurPool2d(64), BlurPool2d(128)
# forward: x = self.bp1(F.max_pool2d(F.relu(self.conv1(x)), 2, stride=1))   # 32 -> 31 -> 16
#          x = self.bp2(F.max_pool2d(F.relu(self.conv2(x)), 2, stride=1))   # 16 -> 15 -> 8
#          x = self.bp3(F.max_pool2d(F.relu(self.conv3(x)), 2, stride=1))   # 8 -> 7 -> 4  (fc1 unchanged)''',
        expected_effect="Paper reports more stable predictions and better robustness to input corruptions on ImageNet models; effect on SimpleCNN is unknown.",
        evidence_level="literature", source="https://arxiv.org/abs/1904.11486 (Zhang, ICML 2019); mapping to our distortion types is a hypothesis"),
    dict(
        id="batchnorm-simplecnn", distortions=["blur", "noise", "jpeg", "pixelate", "mixed"],
        symptom="Low overall accuracy and unstable training after distortion augmentation is added.",
        cause="SimpleCNN has no normalization layers; augmented (harder) data trains slowly and noisily.",
        fix="Add BatchNorm2d after each conv in SimpleCNN.",
        code='''# SimpleCNN.__init__
self.bn1, self.bn2, self.bn3 = torch.nn.BatchNorm2d(32), torch.nn.BatchNorm2d(64), torch.nn.BatchNorm2d(128)
# forward
x = self.pool(F.relu(self.bn1(self.conv1(x))))
x = self.pool(F.relu(self.bn2(self.conv2(x))))
x = self.pool(F.relu(self.bn3(self.conv3(x))))''',
        expected_effect="Faster, steadier training; may raise clean accuracy.",
        evidence_level="hypothesis", source="internal reasoning; must be confirmed by a retrain-and-score run"),
    dict(
        id="resolution-keep-adaptive-pool", distortions=["blur", "pixelate"],
        symptom="Fine detail is lost early; blur/pixelate failures concentrate on small-detail classes.",
        cause="Three successive 2x poolings shrink 32x32 to 4x4, discarding spatial detail.",
        fix="Drop the last pool and use AdaptiveAvgPool2d(4) so the FC input size is unchanged.",
        code='''# SimpleCNN.forward
x = self.pool(F.relu(self.conv1(x)))      # 16x16
x = self.pool(F.relu(self.conv2(x)))      # 8x8
x = F.relu(self.conv3(x))                 # keep 8x8
x = F.adaptive_avg_pool2d(x, 4)           # 4x4 -> fc1 (128*4*4) unchanged''',
        expected_effect="Possible gain on detail-sensitive classes; unknown for distorted inputs.",
        evidence_level="hypothesis", source="internal reasoning; must be confirmed by a retrain-and-score run"),
    dict(
        id="tag-matched-augmentation-mix", distortions=["blur", "noise", "jpeg", "pixelate", "mixed"],
        symptom="Several distortion clusters of different sizes in the YOLO-tagged misclassifications.",
        cause="Uniform augmentation spends effort on distortions that cause few of the errors.",
        fix="Choose augmentations with probability proportional to each distortion's share of the tagged failures.",
        code='''import random
def sample_aug_name(tag_counts):
    """tag_counts e.g. {"blur": 120, "jpeg": 80, "noise": 200, "pixelate": 40} from the YOLO-tagged report"""
    names, w = zip(*tag_counts.items())
    return random.choices(names, weights=w, k=1)[0]''',
        expected_effect="Targets the dominant fault first; use it to order which fix to try.",
        evidence_level="hypothesis", source="internal reasoning; links the diagnostic report to the training pipeline"),
    dict(
        id="mixed-ambiguous-outliers", distortions=["mixed"],
        symptom="Outlier images in a cluster show two distortions at once or look mislabeled.",
        cause="Real failures often combine distortions; single-distortion augmentation does not cover them.",
        fix="Compose two randomly chosen distortions in sequence; inspect outliers for label noise first.",
        code='''import random
class RandomTwo:
    def __init__(self, ops):
        self.ops = ops
    def __call__(self, img):
        for op in random.sample(self.ops, 2):
            img = op(img)
        return img
# ops = [RandomJPEG(), RandomPixelate(), <PIL-compatible blur op>] (cards jpeg-pil-roundtrip-aug, pixelate-downscale-aug)''',
        expected_effect="Better robustness to compound distortions.",
        evidence_level="hypothesis", source="internal reasoning from the diagnostic report's outlier note"),
    dict(
        id="eval-per-distortion-accuracy", distortions=["blur", "noise", "jpeg", "pixelate", "mixed"],
        symptom="Need to confirm a fix helped the diagnosed distortion and did not hurt clean accuracy.",
        cause="Overall accuracy hides per-distortion changes.",
        fix="Evaluate the same model on clean and distorted copies of the test set and report per-distortion accuracy.",
        code='''import torch

@torch.no_grad()
def accuracy(model, loader, device="cpu"):
    model.eval(); correct = total = 0
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        correct += (pred == y).sum().item(); total += y.numel()
    return correct / total
# build one loader per distortion by applying the distortion transform to the TEST set, then compare before/after''',
        expected_effect="Gives the reward signal: delta accuracy on the diagnosed distortion, clean accuracy as a guard.",
        evidence_level="standard", source="CIFAR-10-C (https://zenodo.org/records/2535967) is the standard external check; own corrupted copies work too"),
]

REQUIRED = ["id", "distortions", "symptom", "cause", "fix", "code", "expected_effect", "evidence_level", "source"]
LEVELS = {"literature", "standard", "hypothesis"}
DISTORTIONS = {"blur", "noise", "jpeg", "pixelate", "mixed"}


def validate():
    seen = set()
    for c in CARDS:
        miss = [k for k in REQUIRED if not c.get(k)]
        assert not miss, f"{c.get('id')}: missing {miss}"
        assert c["id"] not in seen, f"duplicate id {c['id']}"
        seen.add(c["id"])
        assert c["evidence_level"] in LEVELS, c["id"]
        assert set(c["distortions"]) <= DISTORTIONS, c["id"]
        if c["evidence_level"] == "literature":
            assert c["source"].startswith("http"), f"{c['id']}: literature card needs a URL"
        # snippets include fragments (comments/pseudo-placement); parse only the code before first fragment marker
        lines = [l for l in c["code"].splitlines() if not l.lstrip().startswith("#")]
        body = "\n".join(lines)
        try:
            ast.parse(body)
        except SyntaxError:
            try:  # method-body fragments (self./forward) -> wrap in a function
                ast.parse("def _f(self, x):\n" + "\n".join("    " + l for l in lines))
            except SyntaxError as e:
                raise AssertionError(f"{c['id']}: snippet does not parse: {e}")
    return len(CARDS)


def smoke():
    import torch, torch.nn as nn, torch.nn.functional as F
    from PIL import Image
    img = Image.new("RGB", (32, 32), (120, 90, 200))
    by_id = {c["id"]: c for c in CARDS}
    results = {}
    for cid in ["blur-gaussian-aug-torchvision", "noise-gaussian-aug", "jpeg-pil-roundtrip-aug",
                "pixelate-downscale-aug", "augmix-torchvision"]:
        ns = {}
        exec(by_id[cid]["code"].split("# in setup")[0].split("# requires")[0], ns)
        out = ns["train_tf"](img)
        assert tuple(out.shape) == (3, 32, 32), (cid, out.shape)
        results[cid] = "ok"
    ns = {}
    exec(by_id["augmix-jsd-consistency"]["code"], ns)
    a = torch.randn(4, 10)
    assert ns["jsd_consistency"](a, a, a).item() < 1e-6          # identical views -> 0
    assert ns["jsd_consistency"](a, torch.randn(4, 10), torch.randn(4, 10)).item() > 0
    results["augmix-jsd-consistency"] = "ok"
    ns = {}
    exec(by_id["blurpool-antialias-simplecnn"]["code"].split("# SimpleCNN")[0], ns)
    x = torch.randn(2, 32, 32, 32)
    y = ns["BlurPool2d"](32)(F.max_pool2d(x, 2, stride=1))
    assert tuple(y.shape) == (2, 32, 16, 16), y.shape
    z = ns["BlurPool2d"](32)(F.max_pool2d(torch.randn(2, 32, 8, 8), 2, stride=1))
    assert tuple(z.shape) == (2, 32, 4, 4), z.shape
    results["blurpool-antialias-simplecnn"] = "ok (32->16, 8->4)"
    ns = {}
    exec(by_id["tag-matched-augmentation-mix"]["code"], ns)
    assert ns["sample_aug_name"]({"blur": 1, "noise": 0}) == "blur"
    results["tag-matched-augmentation-mix"] = "ok"
    return results


if __name__ == "__main__":
    n = validate()
    out = Path(__file__).with_name("fix_cards.jsonl")
    out.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in CARDS) + "\n", encoding="utf-8")
    print(f"validated {n} cards -> {out.name}")
    levels = {}
    for c in CARDS:
        levels[c["evidence_level"]] = levels.get(c["evidence_level"], 0) + 1
    print("evidence levels:", levels)
    if "--smoke" in sys.argv:
        for k, v in smoke().items():
            print(f"smoke {k}: {v}")
