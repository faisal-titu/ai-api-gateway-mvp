"""
Benchmark: PyTorch vs ONNX Runtime for CLIP inference.
Runs text and image encoding through both engines and prints comparison.

Usage:
    python scripts/benchmark_onnx.py
"""
import os
import time
import numpy as np

NUM_WARMUP = 2
NUM_RUNS = 10


def benchmark_pytorch():
    """Benchmark PyTorch CLIP inference."""
    import torch
    import clip
    from PIL import Image

    print("Loading PyTorch CLIP...")
    model, preprocess = clip.load("ViT-B/32", device="cpu")
    model.eval()

    dummy_image = Image.new("RGB", (640, 480), color="red")
    image_input = preprocess(dummy_image).unsqueeze(0)
    text_input = clip.tokenize(["a beautiful sunset over the ocean"])

    # Warmup
    for _ in range(NUM_WARMUP):
        with torch.no_grad():
            model.encode_text(text_input)
            model.encode_image(image_input)

    # Benchmark text
    text_times = []
    for _ in range(NUM_RUNS):
        t0 = time.perf_counter()
        with torch.no_grad():
            features = model.encode_text(text_input)
            features /= features.norm(dim=-1, keepdim=True)
        text_times.append((time.perf_counter() - t0) * 1000)

    # Benchmark image
    image_times = []
    for _ in range(NUM_RUNS):
        t0 = time.perf_counter()
        with torch.no_grad():
            features = model.encode_image(image_input)
            features /= features.norm(dim=-1, keepdim=True)
        image_times.append((time.perf_counter() - t0) * 1000)

    return {
        "text_mean": np.mean(text_times),
        "text_p50": np.median(text_times),
        "text_p95": np.percentile(text_times, 95),
        "image_mean": np.mean(image_times),
        "image_p50": np.median(image_times),
        "image_p95": np.percentile(image_times, 95),
    }


def benchmark_onnx(onnx_dir):
    """Benchmark ONNX Runtime CLIP inference."""
    import onnxruntime as ort
    import clip
    from PIL import Image
    import torch

    text_path = os.path.join(onnx_dir, "clip_text_encoder.onnx")
    image_path = os.path.join(onnx_dir, "clip_image_encoder.onnx")

    print("Loading ONNX sessions...")
    text_session = ort.InferenceSession(text_path, providers=["CPUExecutionProvider"])
    image_session = ort.InferenceSession(image_path, providers=["CPUExecutionProvider"])

    # Need clip for tokenizer and preprocess only
    _, preprocess = clip.load("ViT-B/32", device="cpu")

    dummy_image = Image.new("RGB", (640, 480), color="red")
    image_input = preprocess(dummy_image).unsqueeze(0).numpy()
    text_input = clip.tokenize(["a beautiful sunset over the ocean"]).numpy()

    # Warmup
    for _ in range(NUM_WARMUP):
        text_session.run(None, {"input_ids": text_input})
        image_session.run(None, {"pixel_values": image_input})

    # Benchmark text
    text_times = []
    for _ in range(NUM_RUNS):
        t0 = time.perf_counter()
        result = text_session.run(None, {"input_ids": text_input})[0]
        result = result / np.linalg.norm(result, axis=-1, keepdims=True)
        text_times.append((time.perf_counter() - t0) * 1000)

    # Benchmark image
    image_times = []
    for _ in range(NUM_RUNS):
        t0 = time.perf_counter()
        result = image_session.run(None, {"pixel_values": image_input})[0]
        result = result / np.linalg.norm(result, axis=-1, keepdims=True)
        image_times.append((time.perf_counter() - t0) * 1000)

    return {
        "text_mean": np.mean(text_times),
        "text_p50": np.median(text_times),
        "text_p95": np.percentile(text_times, 95),
        "image_mean": np.mean(image_times),
        "image_p50": np.median(image_times),
        "image_p95": np.percentile(image_times, 95),
    }


def main():
    onnx_dir = os.path.join(os.path.dirname(__file__), "..", "model", "clip_onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    text_onnx = os.path.join(onnx_dir, "clip_text_encoder.onnx")

    # Step 1: Export if needed
    if not os.path.exists(text_onnx):
        print("=" * 60)
        print("STEP 1: Exporting CLIP to ONNX...")
        print("=" * 60)
        import subprocess
        subprocess.run([
            "python", os.path.join(os.path.dirname(__file__), "export_clip_onnx.py"),
            "--output-dir", onnx_dir,
        ], check=True)
        print()

    # Step 2: Benchmark PyTorch
    print("=" * 60)
    print(f"BENCHMARKING PyTorch ({NUM_RUNS} runs, {NUM_WARMUP} warmup)")
    print("=" * 60)
    pt = benchmark_pytorch()

    # Step 3: Benchmark ONNX
    print()
    print("=" * 60)
    print(f"BENCHMARKING ONNX Runtime ({NUM_RUNS} runs, {NUM_WARMUP} warmup)")
    print("=" * 60)
    ox = benchmark_onnx(onnx_dir)

    # Step 4: Print comparison
    print()
    print("=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    print()
    print(f"{'Metric':<25} {'PyTorch':>12} {'ONNX':>12} {'Speedup':>10}")
    print("-" * 60)

    def row(label, pt_val, ox_val):
        speedup = pt_val / ox_val if ox_val > 0 else 0
        print(f"{label:<25} {pt_val:>10.1f}ms {ox_val:>10.1f}ms {speedup:>8.1f}x")

    row("Text Encode (mean)", pt["text_mean"], ox["text_mean"])
    row("Text Encode (p50)", pt["text_p50"], ox["text_p50"])
    row("Text Encode (p95)", pt["text_p95"], ox["text_p95"])
    print()
    row("Image Encode (mean)", pt["image_mean"], ox["image_mean"])
    row("Image Encode (p50)", pt["image_p50"], ox["image_p50"])
    row("Image Encode (p95)", pt["image_p95"], ox["image_p95"])
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
