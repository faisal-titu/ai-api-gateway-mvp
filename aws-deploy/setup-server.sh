#!/bin/bash
# =============================================================
# AWS EC2 Server Setup Script for AI Search API
# Run this ONCE after SSH-ing into your new EC2 instance
# Usage: chmod +x setup-server.sh && sudo ./setup-server.sh
# =============================================================

set -e  # Exit on any error

echo "========================================="
echo "  AI Search API - AWS Server Setup"
echo "========================================="

# ---- 1. System Updates ----
echo "[1/6] Updating system packages..."
dnf update -y

# ---- 2. Install Docker ----
echo "[2/6] Installing Docker..."
dnf install -y docker git
systemctl start docker
systemctl enable docker
usermod -aG docker ec2-user

# ---- 3. Install Docker Compose ----
echo "[3/6] Installing Docker Compose..."
DOCKER_COMPOSE_VERSION="v2.24.5"
curl -SL "https://github.com/docker/compose/releases/download/${DOCKER_COMPOSE_VERSION}/docker-compose-linux-x86_64" \
    -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
ln -sf /usr/local/bin/docker-compose /usr/bin/docker-compose

# Verify installation
docker --version
docker-compose --version

# ---- 4. Create 2GB Swap File ----
echo "[4/6] Creating 2GB swap file..."
if [ ! -f /swapfile ]; then
    dd if=/dev/zero of=/swapfile bs=1M count=2048
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
    echo "Swap file created and enabled."
else
    echo "Swap file already exists, skipping."
fi

# ---- 5. Kernel Settings for OpenSearch ----
echo "[5/6] Setting kernel parameters for OpenSearch..."
sysctl -w vm.max_map_count=262144
echo 'vm.max_map_count=262144' >> /etc/sysctl.conf

# Also lower swappiness to prefer RAM over swap when possible
sysctl -w vm.swappiness=10
echo 'vm.swappiness=10' >> /etc/sysctl.conf

# ---- 6. Create project directory ----
echo "[6/6] Setting up project directory..."
mkdir -p /home/ec2-user/ai-search-api
chown ec2-user:ec2-user /home/ec2-user/ai-search-api

echo ""
echo "========================================="
echo "  Setup Complete!"
echo "========================================="
echo ""
echo "NEXT STEPS (run as ec2-user, NOT root):"
echo "  1. Log out and back in (so Docker group takes effect):"
echo "     exit"
echo "     ssh -i your-key.pem ec2-user@<your-ip>"
echo ""
echo "  2. Clone your repo:"
echo "     cd /home/ec2-user/ai-search-api"
echo "     git clone <your-repo-url> ."
echo ""
echo "  3. Download CLIP model (if not in repo):"
echo "     python3 model/download_models.py"
echo ""
echo "  4. Deploy:"
echo "     docker-compose -f docker-compose.aws.yml up -d --build"
echo ""
echo "  5. Test:"
echo "     curl http://localhost:8000/"
echo "     curl http://localhost:9200/"
echo ""
echo "========================================="
