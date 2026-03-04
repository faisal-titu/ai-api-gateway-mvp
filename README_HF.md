# 🚀 Deploying to Hugging Face Spaces (Free)

This guide explains how to deploy the AI Image Search Engine to **Hugging Face Spaces** for **$0/month**.

## Architecture

| Component | Service | Cost |
|---|---|---|
| Compute | HF Spaces (Docker, 2 vCPU, 16GB RAM) | Free |
| Vector Search | FAISS (in-memory, ~50MB) | Free |
| Image Storage | AWS S3 (`ai-image-searching` bucket) | ~$0.05/month |
| Keep-Alive | UptimeRobot (ping every 5 min) | Free |
| **Total** | | **~$0.05/month** |

---

## Step 1: Create a Hugging Face Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Sign in with your GitHub account
3. Fill in:
   - **Owner**: your username
   - **Space name**: `ai-image-search`
   - **License**: MIT
   - **SDK**: Docker
   - **Hardware**: CPU basic (free)
4. Click **Create Space**

## Step 2: Clone the Space Repository

```bash
# Clone the empty HF Space repo
git clone https://huggingface.co/spaces/YOUR_USERNAME/ai-image-search
cd ai-image-search
```

## Step 3: Copy Project Files

```bash
# From your project directory, copy the necessary files:
cp -r /path/to/ai-api-gateway-mvp/app ./app
cp -r /path/to/ai-api-gateway-mvp/frontend ./frontend
cp -r /path/to/ai-api-gateway-mvp/scripts ./scripts
cp -r /path/to/ai-api-gateway-mvp/dev ./dev
cp -r /path/to/ai-api-gateway-mvp/model ./model
cp /path/to/ai-api-gateway-mvp/requirements.txt .
cp /path/to/ai-api-gateway-mvp/Dockerfile.hf ./Dockerfile
cp /path/to/ai-api-gateway-mvp/embeddings.jsonl .
```

> ⚠️ **Important**: The Dockerfile must be named exactly `Dockerfile` (not `Dockerfile.hf`) in the HF repo.

## Step 4: Install Git LFS (for large files)

The ONNX models and embeddings are large files. HF uses Git LFS for these:

```bash
# Install git-lfs if not already installed
sudo apt install git-lfs
git lfs install

# Track large files
git lfs track "*.onnx"
git lfs track "embeddings.jsonl"
git lfs track "model/clip/*.pt"
```

## Step 5: Push to Hugging Face

```bash
git add .
git commit -m "Initial deployment: AI Image Search Engine"
git push
```

The Space will automatically build and deploy your Docker container. Check the **Logs** tab for build/startup progress.

## Step 6: Set Up UptimeRobot (Keep-Alive)

1. Go to [uptimerobot.com](https://uptimerobot.com) and create a free account
2. Click **Add New Monitor**
3. Configure:
   - **Monitor Type**: HTTP(s)
   - **Friendly Name**: AI Image Search
   - **URL**: `https://YOUR_USERNAME-ai-image-search.hf.space/health`
   - **Monitoring Interval**: 5 minutes
4. Click **Create Monitor**

This prevents the Space from sleeping.

---

## Verifying the Deployment

```bash
# Health check
curl https://YOUR_USERNAME-ai-image-search.hf.space/health

# Text search
curl -X POST "https://YOUR_USERNAME-ai-image-search.hf.space/texts/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "sunset over mountains", "num_images": 5}'

# Admin status
curl https://YOUR_USERNAME-ai-image-search.hf.space/admin/status
```

---

## Updating the Deployment

To push updates, simply commit and push to the HF repo:

```bash
git add .
git commit -m "Update: description of change"
git push
```

The Space rebuilds automatically on each push.

---

## Comparison with AWS Deployment

| Metric | AWS (EC2 + S3) | HF Spaces (FAISS) |
|---|---|---|
| Monthly Cost | ~$31 | **~$0.05** |
| Text Search (warm) | ~500ms | **~210ms** |
| Image Search (warm) | ~1.1s | **~850ms** |
| Always On | ✅ | ✅ (with UptimeRobot) |
| Cold Start | ~80s | ~60-90s |
