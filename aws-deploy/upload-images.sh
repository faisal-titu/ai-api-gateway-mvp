#!/bin/bash
# =============================================================
# Upload dataset images to your AWS EC2 instance
# Run this from YOUR LOCAL MACHINE (not the EC2 server)
#
# Usage:
#   chmod +x upload-images.sh
#   ./upload-images.sh <path-to-pem-key> <elastic-ip>
#
# Example:
#   ./upload-images.sh ~/Downloads/ai-search-key.pem 54.123.45.67
# =============================================================

set -e

PEM_KEY="$1"
SERVER_IP="$2"

if [ -z "$PEM_KEY" ] || [ -z "$SERVER_IP" ]; then
    echo "Usage: ./upload-images.sh <path-to-pem-key> <elastic-ip>"
    echo "Example: ./upload-images.sh ~/Downloads/ai-search-key.pem 54.123.45.67"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_DIR="$PROJECT_DIR/datalake/unsplash"

if [ ! -d "$IMAGE_DIR" ]; then
    echo "Error: Image directory not found at $IMAGE_DIR"
    exit 1
fi

IMAGE_COUNT=$(find "$IMAGE_DIR" -type f | wc -l)
echo "========================================="
echo "  Uploading $IMAGE_COUNT images to AWS"
echo "========================================="
echo "Source: $IMAGE_DIR"
echo "Destination: ec2-user@$SERVER_IP:~/ai-search-api/datalake/unsplash/"
echo ""

# Step 1: Create the directory on the server
echo "[1/3] Creating directory on server..."
ssh -i "$PEM_KEY" ec2-user@"$SERVER_IP" "mkdir -p ~/ai-search-api/datalake/unsplash"

# Step 2: Compress and upload (much faster than scp for many small files)
echo "[2/3] Compressing images..."
cd "$PROJECT_DIR"
tar czf /tmp/unsplash-images.tar.gz -C datalake unsplash/

ARCHIVE_SIZE=$(du -h /tmp/unsplash-images.tar.gz | cut -f1)
echo "       Archive size: $ARCHIVE_SIZE"

echo "[3/3] Uploading to server (this may take a while depending on your internet speed)..."
scp -i "$PEM_KEY" /tmp/unsplash-images.tar.gz ec2-user@"$SERVER_IP":/tmp/

echo "       Extracting on server..."
ssh -i "$PEM_KEY" ec2-user@"$SERVER_IP" "cd ~/ai-search-api/datalake && tar xzf /tmp/unsplash-images.tar.gz && rm /tmp/unsplash-images.tar.gz"

# Cleanup local temp
rm /tmp/unsplash-images.tar.gz

echo ""
echo "========================================="
echo "  Upload Complete!"
echo "========================================="
echo ""
echo "Images are now at: ~/ai-search-api/datalake/unsplash/ on your server"
echo ""
echo "Next: Index the images by running:"
echo "  curl -X POST 'http://$SERVER_IP:8000/images/batch-index?index_name=unsplash_images&image_dir=/app/datalake/unsplash'"
echo ""
