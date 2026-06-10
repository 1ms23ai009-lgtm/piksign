# Image Immunizer Dashboard

A single-GPU web dashboard for M-Attack-V2. Users upload a photo, the server
applies an imperceptible protective perturbation aligned to one **fixed,
preloaded target**, and they download the immunized image.

This is the productized version of the Colab notebook: same proven 224×224 path
(`clip_b16` / `clip_b32` / `clip_laion_b32`, `target_num=1`, no retrieval), but
the 3 models are loaded **once** at startup and kept warm — not reloaded per
request like the CLI does.

## Requirements

- A machine with an **NVIDIA GPU** (a 16 GB T4 is enough, matching the notebook).
  CPU will technically run but is far too slow for real use.
- Python 3.10–3.12.

## Setup

```bash
cd M-Attack-V2
pip install -r dashboard/requirements.txt

# 1) Put your fixed target image here:
#    dashboard/assets/target.png
#    (or point TARGET_PATH at any file)

# 2) Start the server
python dashboard/app.py
```

Open <http://localhost:8000>.

## How it works

- **Startup:** loads the 3 CLIP surrogates onto `cuda:0`, builds the ensemble
  alignment loss, and preloads + preprocesses the fixed target image.
- **Per upload:** preprocesses the source to 224×224, runs `pgd_multi_pass`
  (the M-Attack-V2 attack) against the warm models, returns a lossless PNG.
- **Concurrency:** one GPU job at a time (serialized by a lock); the HTTP server
  stays responsive because the attack runs in a worker thread.

## Knobs exposed in the UI

| Control | Meaning | Trade-off |
|---|---|---|
| **Strength (epsilon)** | L∞ perturbation budget (`/255`) | lower = less visible noise, weaker protection |
| **Steps** | optimization iterations | more = stronger, slower |

Defaults (`epsilon=16`, `steps=300`, `multi_pass=10`) reproduce the notebook.

## Configuration (env vars)

| Var | Default | Notes |
|---|---|---|
| `TARGET_PATH` | `dashboard/assets/target.png` | the fixed target image |
| `PORT` | `8000` | HTTP port |

## Notes & limits

- Output is **224×224** (the proven path). The full-resolution / mid-frequency
  upgrade we scoped is a separate, larger change.
- The protection raises the cost of unauthorized AI editing; it is not an
  absolute guarantee. Use it on images you own.
- Always download/serve the result as the lossless PNG — re-encoding to JPEG or
  resizing afterward weakens the perturbation.
```
