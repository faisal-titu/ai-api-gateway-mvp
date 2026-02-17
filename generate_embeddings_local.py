#!/usr/bin/env python3
"""
Generate CLIP embeddings locally (fast!) and save to JSON.
Then upload to EC2 and bulk-index into OpenSearch.

Usage:
    python generate_embeddings_local.py --image-dir ./datalake/unsplash --output embeddings.jsonl
"""

import os
import sys
import json
import argparse
import time

import clip
import torch
from PIL import Image
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Generate CLIP embeddings locally")
    parser.add_argument("--image-dir", required=True, help="Directory containing images")
    parser.add_argument("--output", default="embeddings.jsonl", help="Output JSONL file")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for CLIP inference")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
    args = parser.parse_args()

    # Check image directory
    if not os.path.isdir(args.image_dir):
        print(f"Error: {args.image_dir} is not a directory")
        sys.exit(1)

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("Loading CLIP model...")
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    print("CLIP model loaded!")

    # List all images
    image_files = sorted([
        f for f in os.listdir(args.image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    print(f"Found {len(image_files)} images")

    # Check for resume
    already_done = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output, "r") as f:
            for line in f:
                data = json.loads(line)
                already_done.add(data["image_id"])
        print(f"Resuming: {len(already_done)} already processed")
        image_files = [f for f in image_files if os.path.splitext(f)[0] not in already_done]
        print(f"Remaining: {len(image_files)} images")

    # Process in batches
    total = len(image_files)
    processed = 0
    errors = 0
    start_time = time.time()

    mode = "a" if args.resume else "w"
    with open(args.output, mode) as out_f:
        for batch_start in tqdm(range(0, total, args.batch_size), desc="Generating embeddings"):
            batch_files = image_files[batch_start:batch_start + args.batch_size]
            images = []
            valid_ids = []

            for fname in batch_files:
                try:
                    img_path = os.path.join(args.image_dir, fname)
                    image = preprocess(Image.open(img_path).convert("RGB"))
                    images.append(image)
                    valid_ids.append(os.path.splitext(fname)[0])
                except Exception as e:
                    errors += 1

            if not images:
                continue

            # Batch inference (much faster than one-by-one)
            image_tensor = torch.stack(images).to(device)
            with torch.no_grad():
                embeddings = model.encode_image(image_tensor)
                embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings_list = embeddings.cpu().tolist()

            for image_id, embedding in zip(valid_ids, embeddings_list):
                out_f.write(json.dumps({
                    "image_id": image_id,
                    "embedding": embedding
                }) + "\n")
                processed += 1

            # Free GPU memory
            del image_tensor, embeddings, images
            if device == "cuda":
                torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    rate = processed / elapsed if elapsed > 0 else 0
    print(f"\nDone! Processed {processed} images in {elapsed:.1f}s ({rate:.1f} images/sec)")
    print(f"Errors: {errors}")
    print(f"Output: {args.output} ({os.path.getsize(args.output) / 1024 / 1024:.1f} MB)")
    print(f"\nNext steps:")
    print(f"  1. Upload to EC2:  scp -i your-key.pem {args.output} ec2-user@<IP>:~/ai-api-gateway-mvp/")
    print(f"  2. Index on EC2:   curl -X POST 'http://localhost:8000/embeddings/bulk-index?index_name=unsplash_images&file_path=/app/{args.output}'")


if __name__ == "__main__":
    main()
