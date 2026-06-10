# Deploying the Immunizer on RunPod

Recommended GPU: **RTX 3090 (24 GB), Community Cloud** — best cost-per-image.
Cheapest fallback: **RTX A4000 (16 GB)**. Avoid Secure Cloud / A100 / L40S (pricier,
unnecessary for the 224px attack).

> 💡 **Stop the pod when idle** — it bills for every hour it's running.

---

## Path A — Custom Docker image (recommended, reproducible)

Models and your target image are baked in, so pod start is fast and identical
every time.

### 1. Build & push the image (from the repo root, on your machine)

```bash
cd M-Attack-V2
docker build -f dashboard/Dockerfile -t YOURUSER/immunizer:latest .
docker push YOURUSER/immunizer:latest
```

(`YOURUSER` = your Docker Hub username; `docker login` first. Image is a few GB
because the CLIP weights are baked in.)

### 2. Create the pod on RunPod

1. RunPod → **Pods → Deploy** → Community Cloud → pick **RTX 3090**.
2. **Container Image:** `YOURUSER/immunizer:latest`
3. **Expose HTTP Port:** `8000`
4. (Optional) Container disk 20 GB is plenty.
5. Deploy.

### 3. Open it

RunPod gives each exposed port a public URL like:

```
https://<POD_ID>-8000.proxy.runpod.net
```

Open that — you get the upload dashboard. Done.

---

## Path B — No Docker build (quick test)

Use RunPod's official **PyTorch** template (CUDA + torch preinstalled), then in
the pod's web terminal:

```bash
git clone https://github.com/VILA-Lab/M-Attack-V2 && cd M-Attack-V2
# apply the same dataclass + dashboard files (or clone your fork that has them)
pip install -r dashboard/requirements.txt
# upload your target to dashboard/assets/target.png
TARGET_PATH=dashboard/assets/target.png python dashboard/app.py
```

Expose port `8000` on the pod and use the proxy URL as above. Slower to start
(downloads models on first run) and manual, but no image build needed.

---

## Sanity checks

```bash
# from your laptop, against the pod URL:
curl https://<POD_ID>-8000.proxy.runpod.net/health
# -> {"status":"ready","device":"cuda:0","backbones":[...],"target_loaded":true}
```

If `status` is `ready` and `target_loaded` is `true`, upload a photo on the page.

## Cost control

- **Stop** (not just close) the pod when you're not using it.
- A 3090 left running ≈ a few $/day; per protected image is only cents.
- If usage grows and you want auto scale-to-zero, move to **RunPod Serverless**
  with the same Docker image (handler wraps the same attack call).
