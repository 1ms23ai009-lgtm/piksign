"""
M-Attack-V2 image-immunization dashboard.

A single-GPU FastAPI service that:
  1. Loads the 3 CLIP surrogate models ONCE at startup (kept warm in memory).
  2. Preloads ONE fixed target image and keeps it ready on the GPU.
  3. Serves a web page where a user uploads a source photo, the protection
     runs, and they download the immunized image.

This reuses the proven 224x224 path from the Colab notebook
(backbones clip_b16 / clip_b32 / clip_laion_b32, target_num=1, no retrieval).

Run (on a machine with an NVIDIA GPU):

    cd M-Attack-V2
    pip install -r dashboard/requirements.txt
    # put your fixed target image at dashboard/assets/target.png  (or set TARGET_PATH)
    python dashboard/app.py

Then open http://localhost:8000
"""

import io
import os
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

# Keep wandb from trying to log/init anything (utils.py imports it).
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")

# Make the repo root importable when launched as `python dashboard/app.py`.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torchvision
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from PIL import Image

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse

from attack import AttackFramework
from surrogates.FeatureExtractors import ClipFeatureExtractor
from surrogates.loss import EnsWeightedMultiAlignmentLoss

# --------------------------------------------------------------------------- #
# Configuration (fixed-target, proven 224px settings)
# --------------------------------------------------------------------------- #
BACKBONES = [
    "clip_b16",
    "clip_b32",
    "clip_laion_b32",
]  # 3-model ensemble (no g14) — ~2-3x faster per image
INPUT_RES = 224
# Strategy: run the attack at the PROVEN working resolution (WORK_RES), then
# smoothly upscale the resulting perturbation onto the full-size original.
#   - WORK_RES (default 224): where the attack is known to transfer. The
#     perturbation stays LOW-FREQUENCY, so it survives the downsampling that
#     ChatGPT/Gemini apply to uploads (high-freq native-res noise does NOT).
#   - Output keeps the ORIGINAL dimensions (long edge capped to MAX_SIDE).
#   - Smooth (bicubic) upscaling of the perturbation looks cleaner than the old
#     blocky 224 output.
WORK_RES = int(os.environ.get("WORK_RES", "224"))  # proven working grid (coarser = survives downsample)
MAX_SIDE = int(os.environ.get("MAX_SIDE", "4096"))
# How the perturbation reaches full size (env-tunable, no rebuild needed):
#   "delta" : scale only the perturbation onto the pristine full-size original
#             (sharper image; perturbation stays low-frequency).
#   "image" : upscale the whole WORK_RES adversarial image (softer image, but
#             closest to the proven-working input -> safest for the effect).
UPSCALE_MODE = os.environ.get("UPSCALE_MODE", "delta").lower()
# JND / contrast masking: in FLAT regions (where the eye sees noise most), keep
# only JND_FLOOR x the perturbation; in TEXTURED regions keep full strength.
#   JND_FLOOR = 1.0  -> no masking == exact current working behaviour (fallback)
#   JND_FLOOR < 1.0  -> flat areas get cleaner (lower = cleaner but riskier)
JND_FLOOR = float(os.environ.get("JND_FLOOR", "1.0"))  # 1.0 = off (isolating consistency)
# EOT over downsampling: during the attack, randomly downsample then restore so
# the perturbation must survive resizing (what ChatGPT/Gemini do to uploads).
# This is what makes the effect CONSISTENT across images, not just easy ones.
#   1.0 = off (exact current behaviour) ; 0.5 = train against up-to-2x downsample
EOT_MIN_SCALE = float(os.environ.get("EOT_MIN_SCALE", "0.5"))
TARGET_PATH = os.environ.get(
    "TARGET_PATH", os.path.join(os.path.dirname(__file__), "assets", "target.png")
)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
USE_AMP = torch.cuda.is_available()

# Attack defaults (match the notebook / ensemble_3models.yaml).
DEFAULTS = {
    "alpha": float(os.environ.get("ALPHA", "0.005")),
    "epsilon": int(os.environ.get("EPSILON", "16")),    # required for the effect to transfer
    "steps": int(os.environ.get("STEPS", "300")),        # restored: effect needs enough iterations
    "optimizer": "adam",
    "momentum": 0.9,
    "momentum_decay": 0.9,
    "beta": 0.3,
    "multi_pass_num": int(os.environ.get("MULTI_PASS", "10")),  # higher = stronger transfer
    "attack": "pgd_multi_pass",
}
print(
    f"[config] epsilon={DEFAULTS['epsilon']} steps={DEFAULTS['steps']} "
    f"multi_pass={DEFAULTS['multi_pass_num']} work_res={WORK_RES} "
    f"jnd_floor={JND_FLOOR} eot_min_scale={EOT_MIN_SCALE}"
)
app = FastAPI(title="Image Shield")

# --- Async job system -------------------------------------------------------
# A single image takes minutes, but RunPod's HTTP proxy (and browsers) drop a
# request after ~100s. So /protect returns a job id immediately and the page
# polls /status, then downloads from /result. max_workers=1 => one GPU job at a
# time; extra submissions queue automatically.
_executor = ThreadPoolExecutor(max_workers=1)
_jobs = {}  # job_id -> {status, result(bytes), error, elapsed, filename}
_jobs_lock = threading.Lock()

# Warm globals, populated at startup.
_models = None
_ensemble_loss = None
_target_tensor = None  # [1, 3, 224, 224] on DEVICE
_source_crop = None
_target_crop = None
_change_iters = [150, 275]  # proven single-GPU schedule

_preprocess = transforms.Compose(
    [
        transforms.Resize(
            INPUT_RES, interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.CenterCrop(INPUT_RES),
        transforms.ToTensor(),  # -> [0, 1] float
    ]
)  # used for the TARGET (stays 224)

_to_tensor = transforms.ToTensor()


def _load_source_native(img: Image.Image) -> torch.Tensor:
    """Source -> tensor at native resolution (long edge capped to MAX_SIDE).

    Preserves the original dimensions/aspect ratio so the protected output is
    the same size as the upload; only downscales if larger than MAX_SIDE.
    """
    w, h = img.size
    long_edge = max(w, h)
    if long_edge > MAX_SIDE:
        scale = MAX_SIDE / long_edge
        img = img.resize(
            (max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC
        )
    return _to_tensor(img)


def _jnd_mask(img: torch.Tensor) -> torch.Tensor:
    """Per-pixel perceptual mask in [JND_FLOOR, 1].

    ~1 in textured regions (noise is hidden there), -> JND_FLOOR in flat regions
    (sky, walls, skin — where the eye notices noise most). Uses local luminance
    standard deviation as a contrast-masking proxy. Shape [1,1,H,W], broadcasts
    over the 3 colour channels of the perturbation.
    """
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    k, pad = 7, 3
    mean = torch.nn.functional.avg_pool2d(lum, k, stride=1, padding=pad)
    mean_sq = torch.nn.functional.avg_pool2d(lum * lum, k, stride=1, padding=pad)
    std = (mean_sq - mean * mean).clamp(min=0.0).sqrt()
    scale = std.mean() * 2.0 + 1e-6
    activity = (std / scale).clamp(0.0, 1.0)
    return JND_FLOOR + (1.0 - JND_FLOOR) * activity


class _RandomDownsampleEOT:
    """EOT over resolution: randomly downsample the crop then restore it, so the
    perturbation is forced to survive resizing (like the black-box's resize of an
    upload). Identity when min_scale >= 1.0 (i.e. EOT off)."""

    def __init__(self, res: int, min_scale: float, max_scale: float = 1.0):
        self.res = res
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.min_scale >= 0.999:
            return x
        f = float(torch.empty(1).uniform_(self.min_scale, self.max_scale).item())
        low = max(16, int(round(self.res * f)))
        x = torch.nn.functional.interpolate(
            x, size=(low, low), mode="bilinear", align_corners=False, antialias=True
        )
        x = torch.nn.functional.interpolate(
            x, size=(self.res, self.res), mode="bilinear", align_corners=False
        )
        return x


def _build_cfg(epsilon: int, steps: int, multi_pass_num: int) -> OmegaConf:
    """Construct the minimal config the attack code reads from."""
    return OmegaConf.create(
        {
            "optim": {
                "alpha": DEFAULTS["alpha"],
                "epsilon": int(epsilon),
                "steps": int(steps),
                "optimizer": DEFAULTS["optimizer"],
                "momentum": DEFAULTS["momentum"],
                "momentum_decay": DEFAULTS["momentum_decay"],
                "beta": DEFAULTS["beta"],
                "multi_pass_num": int(multi_pass_num),
            },
            "model": {"input_res": INPUT_RES},
            "attack": DEFAULTS["attack"],
        }
    )


@app.on_event("startup")
def _load_everything() -> None:
    global _models, _ensemble_loss, _target_tensor, _source_crop, _target_crop

    print(f"[startup] device={DEVICE}  amp={USE_AMP}")
    print(f"[startup] loading backbones {BACKBONES} (one-time)...")
    t0 = time.time()
    _models = [
        ClipFeatureExtractor(name).eval().to(DEVICE).requires_grad_(False)
        for name in BACKBONES
    ]
    _ensemble_loss = EnsWeightedMultiAlignmentLoss(_models, beta=DEFAULTS["beta"])
    print(f"[startup] models ready in {time.time() - t0:.1f}s")

    if not os.path.exists(TARGET_PATH):
        raise RuntimeError(
            f"Fixed target image not found at '{TARGET_PATH}'. "
            "Place your target there or set the TARGET_PATH env var."
        )
    tgt = Image.open(TARGET_PATH).convert("RGB")
    _target_tensor = _preprocess(tgt).unsqueeze(0).to(DEVICE)
    print(f"[startup] preloaded fixed target from {TARGET_PATH}")

    _eot = _RandomDownsampleEOT(INPUT_RES, EOT_MIN_SCALE)
    _source_crop = [
        transforms.Compose(
            [transforms.RandomResizedCrop(INPUT_RES, scale=(0.5, 1.0)), _eot]
        )
        for _ in range(3)
    ]
    _target_crop = transforms.Compose(
        [
            transforms.RandomResizedCrop(INPUT_RES, scale=(0.95, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),
        ]
    )
    print("[startup] ready to serve.")


def _run_attack(image_org, epsilon, steps, multi_pass_num):
    """Synchronous GPU work. Serialized by _gpu_lock."""
    cfg = _build_cfg(epsilon, steps, multi_pass_num)
    attacker = AttackFramework.create(
        attack_type=DEFAULTS["attack"],
        cfg=cfg,
        ensemble_loss=_ensemble_loss,
        source_crop=_source_crop,
        target_crop=_target_crop,
        change_iters=_change_iters,
    )
    adv = attacker.attack(
        img_index=0,
        image_org=image_org,
        image_tgt=_target_tensor,
        target_num=1,
        amp=USE_AMP,
        log_wandb=False,
        log_interval=10 ** 9,  # effectively never log to a pbar/wandb
        disable_tqdm=True,
        multi_pass_num=int(multi_pass_num),
    )
    return adv.detach()


def _job_worker(job_id: str, raw: bytes) -> None:
    """Run one protection job (background thread, serialized by the executor)."""
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"
    try:
        src = Image.open(io.BytesIO(raw)).convert("RGB")
        src_native = _load_source_native(src).unsqueeze(0).to(DEVICE)  # [1,3,H,W]
        h, w = src_native.shape[-2:]
        # Run the attack at the proven working resolution.
        src_work = torch.nn.functional.interpolate(
            src_native, size=(WORK_RES, WORK_RES), mode="bicubic", align_corners=False
        ).clamp(0.0, 1.0)
        t0 = time.time()
        adv_work = _run_attack(
            src_work,
            DEFAULTS["epsilon"],
            DEFAULTS["steps"],
            DEFAULTS["multi_pass_num"],
        )
        elapsed = time.time() - t0

        # Compose the full-size output, keeping the original dimensions.
        if UPSCALE_MODE == "image":
            adv_native = torch.nn.functional.interpolate(
                adv_work, size=(h, w), mode="bicubic", align_corners=False
            ).clamp(0.0, 1.0)
        else:  # "delta" (default): scale only the (low-freq) perturbation up
            delta_native = torch.nn.functional.interpolate(
                adv_work - src_work, size=(h, w), mode="bicubic", align_corners=False
            )
            if JND_FLOOR < 1.0:  # hide noise in flat regions (keep it in texture)
                delta_native = delta_native * _jnd_mask(src_native)
            adv_native = (src_native + delta_native).clamp(0.0, 1.0)

        buf = io.BytesIO()
        torchvision.utils.save_image(adv_native[0].cpu(), buf, format="png")
        with _jobs_lock:
            _jobs[job_id].update(
                status="done", result=buf.getvalue(), elapsed=round(elapsed, 1)
            )
        print(
            f"[protect] {_jobs[job_id]['filename']} {w}x{h} "
            f"mode={UPSCALE_MODE} work={WORK_RES} done in {elapsed:.1f}s"
        )
    except Exception as exc:  # noqa: BLE001
        with _jobs_lock:
            _jobs[job_id].update(status="error", error=str(exc))
        print(f"[protect] job {job_id} failed: {exc}")


@app.post("/protect")
async def protect(file: UploadFile = File(...)):
    # Parameters are FIXED server-side; users cannot change them (UI or raw API).
    raw = await file.read()
    try:
        Image.open(io.BytesIO(raw)).verify()  # validate it's a real image
    except Exception:
        raise HTTPException(
            status_code=400, detail="That doesn't look like a valid image."
        )

    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "result": None,
            "error": None,
            "elapsed": None,
            "filename": os.path.basename(file.filename or "image"),
        }
        ahead = sum(1 for j in _jobs.values() if j["status"] in ("queued", "running")) - 1
    _executor.submit(_job_worker, job_id, raw)
    return {"job_id": job_id, "ahead": max(0, ahead)}


@app.get("/status/{job_id}")
def status(job_id: str):
    with _jobs_lock:
        j = _jobs.get(job_id)
        if j is None:
            raise HTTPException(status_code=404, detail="Unknown job.")
        return {"status": j["status"], "elapsed": j["elapsed"], "error": j["error"]}


@app.get("/result/{job_id}")
def result(job_id: str):
    with _jobs_lock:
        j = _jobs.get(job_id)
    if j is None or j["status"] != "done":
        raise HTTPException(status_code=404, detail="Result not ready.")
    name = os.path.splitext(j["filename"])[0]
    return StreamingResponse(
        io.BytesIO(j["result"]),
        media_type="image/png",
        headers={
            "Content-Disposition": f'attachment; filename="{name}_protected.png"'
        },
    )


@app.get("/health")
def health():
    return {
        "status": "ready" if _models is not None else "loading",
        "device": DEVICE,
        "backbones": BACKBONES,
        "target_loaded": _target_tensor is not None,
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<link rel="icon" href="data:," />
<title>PikSign Protection</title>
<style>
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif;
    margin: 0; min-height: 100vh; display: flex; align-items: center; justify-content: center;
    padding: 24px; color: #1b1f2a;
    background:
      radial-gradient(1100px 560px at 8% -12%, #ece7ff 0%, transparent 60%),
      radial-gradient(900px 520px at 112% 114%, #ffe6f3 0%, transparent 55%),
      linear-gradient(135deg, #f5f4ff 0%, #fdf3fb 100%);
  }
  .card {
    background: rgba(255,255,255,.92); backdrop-filter: blur(10px);
    border: 1px solid rgba(124,92,255,.10); border-radius: 28px; width: min(470px, 100%);
    box-shadow: 0 30px 70px rgba(80, 50, 150, 0.18); padding: 34px 30px; text-align: center;
  }
  .brand { display: flex; align-items: center; justify-content: center; gap: 11px; }
  .badge {
    width: 42px; height: 42px; border-radius: 13px; display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, #7c5cff, #c45cff); box-shadow: 0 8px 18px rgba(124,92,255,.38);
  }
  .brand-name {
    font-size: 23px; font-weight: 800; letter-spacing: -0.02em;
    background: linear-gradient(135deg, #6d44ff, #c145ff);
    -webkit-background-clip: text; background-clip: text; color: transparent;
  }
  .tagline { color: #6b7280; font-size: 13.5px; margin: 15px auto 24px; line-height: 1.55; max-width: 340px; }

  .drop {
    border: 2px dashed #d3d8e8; border-radius: 18px; padding: 36px 18px; cursor: pointer;
    transition: all .18s ease; background: #fbfaff;
  }
  .drop:hover, .drop.over { border-color: #7c5cff; background: #f3efff; transform: translateY(-1px); }
  .drop .up-ico { display: flex; justify-content: center; }
  .drop .t1 { font-weight: 700; margin-top: 13px; font-size: 15.5px; color: #2a2f3a; }
  .drop .t2 { color: #9aa1b2; font-size: 12.5px; margin-top: 4px; }
  .drop .t3 { color: #b6bccb; font-size: 11.5px; margin-top: 9px; letter-spacing: .04em; }
  .thumb { max-height: 165px; max-width: 100%; border-radius: 13px; margin: 0 auto; display: block; }
  .fname { font-size: 13px; color: #4b5563; margin-top: 11px; word-break: break-all; }
  .change { color: #7c5cff; font-size: 12.5px; margin-top: 7px; font-weight: 600; }

  .btn {
    margin-top: 18px; width: 100%; padding: 15px; font-size: 15px; font-weight: 700; color: #fff;
    border: 0; border-radius: 14px; cursor: pointer; letter-spacing: .01em;
    background: linear-gradient(135deg, #7c5cff, #c145ff);
    box-shadow: 0 10px 24px rgba(124, 92, 255, .34); transition: transform .1s, opacity .15s;
  }
  .btn:hover { transform: translateY(-1px); }
  .btn:disabled { opacity: .45; cursor: not-allowed; transform: none; box-shadow: none; }

  .spinner {
    width: 46px; height: 46px; margin: 8px auto 0; border-radius: 50%;
    border: 4px solid #ece8ff; border-top-color: #7c5cff; animation: spin 0.9s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .working-title { font-weight: 700; margin-top: 18px; font-size: 16px; }
  .timer { font-variant-numeric: tabular-nums; color: #7c5cff; font-weight: 800; font-size: 26px; margin-top: 6px; }
  .note { color: #9aa1b2; font-size: 12.5px; margin-top: 12px; line-height: 1.55; }

  .result-img { max-width: 100%; border-radius: 16px; border: 1px solid #eceef4; margin-top: 4px; }
  .done-badge { color: #16a34a; font-weight: 800; font-size: 16px; margin: 6px 0 14px; }
  .err { color: #dc2626; font-size: 13.5px; margin-top: 14px; line-height: 1.5; }
  .how { margin-top: 24px; padding-top: 16px; border-top: 1px solid #f0f1f6; color: #9aa1b2; font-size: 11.5px; line-height: 1.6; }
  .hidden { display: none; }
  a.linkbtn { display: inline-block; margin-top: 14px; color: #7c5cff; font-weight: 700; font-size: 13.5px; text-decoration: none; cursor: pointer; }
</style>
</head>
<body>
  <div class="card">
    <div class="brand">
      <div class="badge">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="11" rx="2.2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>
      </div>
      <div class="brand-name">PikSign Protection</div>
    </div>
    <p class="tagline">Protect your photos from AI editing. Upload a photo and get back a protected version that looks the same.</p>

    <!-- STATE: upload -->
    <div id="view-upload">
      <div class="drop" id="drop">
        <div id="drop-empty">
          <div class="up-ico">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#7c5cff" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
          </div>
          <div class="t1">Drag &amp; drop your photo here</div>
          <div class="t2">or click to browse</div>
          <div class="t3">JPG · PNG</div>
        </div>
        <div id="drop-preview" class="hidden">
          <img class="thumb" id="thumb" alt="preview" />
          <div class="fname" id="fname"></div>
          <div class="change">Click to choose a different photo</div>
        </div>
      </div>
      <input id="file" type="file" accept="image/*" class="hidden" />
      <button class="btn" id="go" disabled>Protect my photo</button>
    </div>

    <!-- STATE: working -->
    <div id="view-working" class="hidden">
      <div class="spinner"></div>
      <div class="working-title" id="working-title">Protecting your photo…</div>
      <div class="timer" id="timer">0:00</div>
      <div class="note">This usually takes a few minutes.<br>Please keep this tab open.</div>
    </div>

    <!-- STATE: done -->
    <div id="view-done" class="hidden">
      <div class="done-badge">✓ Your photo is protected</div>
      <img class="result-img" id="result-img" alt="protected result" />
      <a class="btn" id="download" style="margin-top:18px; text-decoration:none; display:block;">⬇ Download protected photo</a>
      <a class="linkbtn" id="again">Protect another photo</a>
    </div>

    <!-- STATE: error -->
    <div id="view-error" class="hidden">
      <div class="err" id="err-msg"></div>
      <a class="linkbtn" id="retry">Try again</a>
    </div>

    <div class="how">Upload → we add invisible protection → download. The protected photo looks identical but resists AI manipulation.</div>
  </div>

<script>
const $ = (id) => document.getElementById(id);
const views = ['upload', 'working', 'done', 'error'];
function show(state) {
  views.forEach(v => $('view-' + v).classList.toggle('hidden', v !== state));
}

let selectedFile = null;
const fileInput = $('file');
const drop = $('drop');

drop.onclick = () => fileInput.click();
fileInput.onchange = () => pickFile(fileInput.files[0]);
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add('over'); };
drop.ondragleave = () => drop.classList.remove('over');
drop.ondrop = (e) => {
  e.preventDefault(); drop.classList.remove('over');
  if (e.dataTransfer.files.length) pickFile(e.dataTransfer.files[0]);
};

function pickFile(f) {
  if (!f || !f.type.startsWith('image/')) return;
  selectedFile = f;
  $('thumb').src = URL.createObjectURL(f);
  $('fname').textContent = f.name;
  $('drop-empty').classList.add('hidden');
  $('drop-preview').classList.remove('hidden');
  $('go').disabled = false;
}

let timerInt = null;
function startTimer() {
  const t0 = Date.now();
  const tick = () => {
    const s = Math.floor((Date.now() - t0) / 1000);
    $('timer').textContent = Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0');
  };
  tick();
  timerInt = setInterval(tick, 1000);
}
function stopTimer() { if (timerInt) clearInterval(timerInt); timerInt = null; }

$('go').onclick = async () => {
  if (!selectedFile) return;
  show('working');
  startTimer();
  try {
    const fd = new FormData();
    fd.append('file', selectedFile);
    const r = await fetch('/protect', { method: 'POST', body: fd });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || 'Upload failed');
    const { job_id } = await r.json();
    poll(job_id);
  } catch (err) {
    fail(err.message);
  }
};

function poll(jobId) {
  const iv = setInterval(async () => {
    try {
      const s = await (await fetch('/status/' + jobId)).json();
      if (s.status === 'queued') $('working-title').textContent = 'Waiting in line…';
      else if (s.status === 'running') $('working-title').textContent = 'Protecting your photo…';
      else if (s.status === 'done') { clearInterval(iv); stopTimer(); finish(jobId); }
      else if (s.status === 'error') { clearInterval(iv); fail(s.error || 'Something went wrong'); }
    } catch (e) { /* transient network blip — keep polling */ }
  }, 2500);
}

async function finish(jobId) {
  const blob = await (await fetch('/result/' + jobId)).blob();
  const url = URL.createObjectURL(blob);
  $('result-img').src = url;
  const dl = $('download');
  dl.href = url; dl.download = 'protected.png';
  show('done');
}

function fail(msg) {
  stopTimer();
  $('err-msg').textContent = msg || 'Something went wrong. Please try again.';
  show('error');
}

$('again').onclick = () => resetUI();
$('retry').onclick = () => resetUI();
function resetUI() {
  selectedFile = null; fileInput.value = '';
  $('drop-empty').classList.remove('hidden');
  $('drop-preview').classList.add('hidden');
  $('go').disabled = true;
  show('upload');
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
