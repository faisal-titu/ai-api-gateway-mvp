# AWS Deployment Guide — AI Search API (Free Tier)

> **Difficulty**: Beginner · **Cost**: $0 (AWS Free Tier) · **Time**: ~30 minutes

This guide walks you through deploying the AI Search API on AWS from scratch.

---

## Prerequisites

- ✅ An AWS account (you already have one)
- ✅ A credit/debit card linked to AWS (won't be charged on free tier)
- ✅ Your project code pushed to a Git repository (GitHub/Bitbucket)
- ✅ CLIP model files downloaded in `model/clip/` (run `python model/download_models.py` locally first if not already done)

---

## Step 1: Launch an EC2 Instance

1. Go to **[AWS Console](https://console.aws.amazon.com/)** → search for **EC2** → click **EC2 Dashboard**

2. Click **"Launch Instance"** (orange button)

3. Fill in these settings:

   | Setting | Value |
   |---------|-------|
   | **Name** | `ai-search-api` |
   | **AMI** | Amazon Linux 2023 AMI _(free tier eligible — look for the green label)_ |
   | **Instance type** | `t2.micro` _(free tier eligible)_ |
   | **Key pair** | Click **"Create new key pair"** → name it `ai-search-key` → type: RSA → format: `.pem` → **download it** |
   | **Storage** | Change to **30 GB** `gp3` (max free tier) |

4. Under **Network settings** → click **"Edit"**:
   - ✅ **Allow SSH traffic** (port 22) — from "My IP"
   - Click **"Add security group rule"**:
     - Type: **Custom TCP** → Port: **8000** → Source: **Anywhere (0.0.0.0/0)**
   - Click **"Add security group rule"** again:
     - Type: **Custom TCP** → Port: **9200** → Source: **My IP** _(for OpenSearch, keep restricted)_

5. Click **"Launch Instance"** ✅

> ⚠️ **Keep the `.pem` key file safe!** You need it to connect to your server. Never share it.

---

## Step 2: Allocate an Elastic IP (Fixed Address)

Without an Elastic IP, your server IP changes every time it restarts.

1. In EC2 Dashboard → left sidebar → **"Elastic IPs"**
2. Click **"Allocate Elastic IP address"** → **Allocate**
3. Select the new IP → **Actions** → **"Associate Elastic IP address"**
4. Choose your `ai-search-api` instance → **Associate**

**Note your Elastic IP** — this is your permanent server address.

> 💡 Elastic IPs are free ONLY while associated with a running instance. If you stop the instance, release the Elastic IP to avoid charges.

---

## Step 3: Connect to Your Server via SSH

### On Linux/Mac:
```bash
# Set correct permissions on your key file (required, do this once)
chmod 400 ~/Downloads/ai-search-key.pem

# Connect to your server
ssh -i ~/Downloads/ai-search-key.pem ec2-user@<YOUR_ELASTIC_IP>
```

### On Windows:
Use **PuTTY** or **Windows Terminal**:
```bash
ssh -i C:\Users\YourName\Downloads\ai-search-key.pem ec2-user@<YOUR_ELASTIC_IP>
```

If it asks "Are you sure you want to continue connecting?" → type **yes**

You should see:
```
   ,     #_
   ~\_  ####_        Amazon Linux 2023
  ~~  \_#####\
  ~~     \###|       
  ~~       \#/ ___   
   ~~       V~' '->  
    ~~~         /    
      ~~._.   _/     
         _/ _/       
       _/m/'          
[ec2-user@ip-xxx ~]$
```

---

## Step 4: Run the Setup Script

### Option A: If your repo is public (GitHub/Bitbucket)

```bash
# Clone the repo
cd ~
git clone <YOUR_REPO_URL> ai-search-api
cd ai-search-api

# Run the setup script as root
sudo chmod +x aws-deploy/setup-server.sh
sudo ./aws-deploy/setup-server.sh
```

### Option B: If your repo is private

You'll need to set up SSH keys or use a personal access token:

```bash
# Using HTTPS + personal access token
cd ~
git clone https://<USERNAME>:<TOKEN>@bitbucket.org/resilientsage/ai-api-gateway-mvp.git ai-search-api
cd ai-search-api

# Run setup
sudo chmod +x aws-deploy/setup-server.sh
sudo ./aws-deploy/setup-server.sh
```

**After the setup**:
```bash
# IMPORTANT: Log out and log back in for Docker group to work
exit
ssh -i ~/Downloads/ai-search-key.pem ec2-user@<YOUR_ELASTIC_IP>
```

---

## Step 5: Deploy the Application

```bash
cd ~/ai-search-api

# Verify swap is active (should show ~2GB swap)
free -h

# Build and start the containers (this takes 5-10 minutes on first run)
docker-compose -f docker-compose.aws.yml up -d --build

# Watch the logs to see when it's ready
docker-compose -f docker-compose.aws.yml logs -f
# Wait until you see "Application startup complete" — then press Ctrl+C
```

---

## Step 6: Test Your Deployment

### From the EC2 server itself:
```bash
# Test the API
curl http://localhost:8000/
# Expected: {"message":"Welcome to the AI Search API"}

# Test health check
curl http://localhost:8000/health

# Test OpenSearch
curl http://localhost:9200/
# Expected: OpenSearch cluster info JSON
```

### From your local machine (laptop):
```bash
# Replace <YOUR_ELASTIC_IP> with your actual Elastic IP
curl http://<YOUR_ELASTIC_IP>:8000/
# Expected: {"message":"Welcome to the AI Search API"}
```

🎉 **If you see the welcome message, your API is live!**

---

## Step 7: Upload Your Image Dataset

Your 24,977 Unsplash images (~2.1 GB) are gitignored, so they need to be uploaded separately.

### Option A: Use the upload script (recommended)

**Run this from your LOCAL machine** (not the EC2 server):
```bash
cd /path/to/ai-api-gateway-mvp
chmod +x aws-deploy/upload-images.sh
./aws-deploy/upload-images.sh ~/Downloads/ai-search-key.pem <YOUR_ELASTIC_IP>
```

This compresses the images → uploads via SCP → extracts on the server. Takes ~10-30 mins depending on your upload speed.

### Option B: Manual upload

```bash
# From your local machine — compress first (much faster than scp-ing 25K files)
cd /path/to/ai-api-gateway-mvp
tar czf /tmp/unsplash-images.tar.gz -C datalake unsplash/

# Upload the archive
scp -i ~/Downloads/ai-search-key.pem /tmp/unsplash-images.tar.gz ec2-user@<YOUR_ELASTIC_IP>:/tmp/

# SSH into server and extract
ssh -i ~/Downloads/ai-search-key.pem ec2-user@<YOUR_ELASTIC_IP>
cd ~/ai-search-api/datalake
tar xzf /tmp/unsplash-images.tar.gz
rm /tmp/unsplash-images.tar.gz
```

### After upload — verify on the server:
```bash
ls ~/ai-search-api/datalake/unsplash/ | wc -l
# Expected: 24977
```

> ⚠️ **After uploading, restart the containers** so the volume mount picks up the data:
> ```bash
> docker-compose -f docker-compose.aws.yml restart ai-search-api
> ```

---

## Step 8: Index and Search Your Images

1. **Set the index settings**:
```bash
curl -X POST http://<YOUR_ELASTIC_IP>:8000/set-settings \
  -H "Content-Type: application/json" \
  -d '{"index_name": "unsplash_images", "image_dir": "/app/datalake/unsplash"}'
```

2. **Batch index all images** (this takes a while — 25K images on a t2.micro):
```bash
curl -X POST "http://<YOUR_ELASTIC_IP>:8000/images/batch-index?index_name=unsplash_images&image_dir=/app/datalake/unsplash"
```
> 💡 This runs CLIP embeddings on all images. On `t2.micro` it may take several hours. You can start with a subset for testing.

3. **Search by text**:
```bash
curl -X POST http://<YOUR_ELASTIC_IP>:8000/texts/search \
  -H "Content-Type: application/json" \
  -d '{"query": "a photo of a sunset", "num_images": 5}'
```

4. **Search by image** (upload an image as query):
```bash
curl -X POST http://<YOUR_ELASTIC_IP>:8000/images/search \
  -F "file=@/path/to/your/query-image.jpg" \
  -F "num_images=5"
```

---

## Cost Monitoring (Stay Free!)

### Set up billing alerts:
1. Go to **AWS Console** → search **"Billing"**
2. **Budgets** → **Create a budget**
3. Choose **"Zero spend budget"** → this alerts you if ANY charges appear
4. Enter your email → Create

### Free tier limits to watch:
| Resource | Free Limit | What counts |
|----------|-----------|-------------|
| EC2 | 750 hours/month | Your instance running time |
| EBS | 30 GB | Your disk storage |
| Data transfer | 100 GB out | API responses sent to users |
| Elastic IP | 1 (while attached to running instance) | ⚠️ **Release if you stop the instance!** |

### When you're NOT using it:
```bash
# Stop containers (saves CPU, keeps data)
docker-compose -f docker-compose.aws.yml down

# To fully stop the EC2 instance (stops billing for compute):
# Go to EC2 Console → select instance → Instance State → Stop
# ⚠️ Release Elastic IP first, or it will cost ~$3.60/month while unattached!
```

---

## Troubleshooting

### "Cannot connect to the Docker daemon"
```bash
sudo systemctl start docker
```

### "Out of memory" / container keeps restarting
```bash
# Check memory usage
free -h
docker stats

# If needed, restart with fewer resources
docker-compose -f docker-compose.aws.yml down
docker-compose -f docker-compose.aws.yml up -d
```

### "Connection refused" on port 8000
- Check Security Group in EC2 console — port 8000 must be open
- Check if containers are running: `docker ps`
- Check container logs: `docker-compose -f docker-compose.aws.yml logs ai-search-api`

### OpenSearch won't start
```bash
# Check if the kernel parameter is set
sysctl vm.max_map_count
# Should show 262144. If not:
sudo sysctl -w vm.max_map_count=262144
```

---

## Quick Reference

| Action | Command |
|--------|---------|
| **Start** | `docker-compose -f docker-compose.aws.yml up -d` |
| **Stop** | `docker-compose -f docker-compose.aws.yml down` |
| **View logs** | `docker-compose -f docker-compose.aws.yml logs -f` |
| **Rebuild** | `docker-compose -f docker-compose.aws.yml up -d --build` |
| **Check status** | `docker ps` |
| **Check memory** | `free -h && docker stats --no-stream` |
| **SSH connect** | `ssh -i ai-search-key.pem ec2-user@<IP>` |
