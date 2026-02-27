"""
Export OpenAI CLIP (ViT-B/32) text and image encoders to ONNX format.

Usage:
    python scripts/export_clip_onnx.py --output-dir /app/model/clip_onnx

This produces two files:
    clip_text_encoder.onnx   (~170 MB)
    clip_image_encoder.onnx  (~340 MB)
"""

import os
import argparse
import torch
import clip
import numpy as np


class CLIPTextEncoder(torch.nn.Module):
    """Wrapper that isolates CLIP's text encoder for ONNX export."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)


class CLIPImageEncoder(torch.nn.Module):
    """Wrapper that isolates CLIP's image encoder for ONNX export."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)


def export_text_encoder(model, output_path):
    """Export the text encoder to ONNX."""
    print("Exporting text encoder...")
    text_encoder = CLIPTextEncoder(model)
    text_encoder.eval()

    # Dummy input: tokenized text (batch=1, seq_len=77)
    dummy_text = clip.tokenize(["a photo of a cat"]).to("cpu")

    torch.onnx.export(
        text_encoder,
        dummy_text,
        output_path,
        input_names=["input_ids"],
        output_names=["text_features"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "text_features": {0: "batch_size"}
        },
        opset_version=14,
        do_constant_folding=True,
    )
    print(f"  ✅ Text encoder saved to {output_path}")


def export_image_encoder(model, output_path):
    """Export the image encoder to ONNX."""
    print("Exporting image encoder...")
    image_encoder = CLIPImageEncoder(model)
    image_encoder.eval()

    # Dummy input: preprocessed image (batch=1, channels=3, H=224, W=224)
    dummy_image = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    torch.onnx.export(
        image_encoder,
        dummy_image,
        output_path,
        input_names=["pixel_values"],
        output_names=["image_features"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "image_features": {0: "batch_size"}
        },
        opset_version=14,
        do_constant_folding=True,
    )
    print(f"  ✅ Image encoder saved to {output_path}")


def verify_onnx(model, preprocess, text_onnx_path, image_onnx_path):
    """Verify ONNX models produce same results as PyTorch."""
    import onnxruntime as ort

    print("\nVerifying ONNX accuracy...")

    # --- Text verification ---
    text_input = clip.tokenize(["a sunset over the ocean"]).to("cpu")

    with torch.no_grad():
        pt_text = model.encode_text(text_input).cpu().numpy()

    text_session = ort.InferenceSession(text_onnx_path)
    onnx_text = text_session.run(None, {"input_ids": text_input.numpy()})[0]

    text_diff = np.max(np.abs(pt_text - onnx_text))
    print(f"  Text encoder max diff: {text_diff:.6f} {'✅' if text_diff < 0.01 else '❌'}")

    # --- Image verification ---
    from PIL import Image
    dummy_img = Image.new("RGB", (224, 224), color="blue")
    image_input = preprocess(dummy_img).unsqueeze(0)

    with torch.no_grad():
        pt_image = model.encode_image(image_input).cpu().numpy()

    image_session = ort.InferenceSession(image_onnx_path)
    onnx_image = image_session.run(None, {"pixel_values": image_input.numpy()})[0]

    image_diff = np.max(np.abs(pt_image - onnx_image))
    print(f"  Image encoder max diff: {image_diff:.6f} {'✅' if image_diff < 0.01 else '❌'}")


def main():
    parser = argparse.ArgumentParser(description="Export CLIP to ONNX")
    parser.add_argument("--output-dir", default="/app/model/clip_onnx",
                        help="Directory to save ONNX models")
    parser.add_argument("--model-cache", default="/app/model/clip",
                        help="Path to cached CLIP model weights")
    parser.add_argument("--verify", action="store_true", default=True,
                        help="Verify ONNX output matches PyTorch")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load CLIP
    print(f"Loading CLIP ViT-B/32 from {args.model_cache}...")
    try:
        model, preprocess = clip.load("ViT-B/32", device="cpu",
                                       download_root=args.model_cache)
    except Exception:
        print("  Cache miss, downloading...")
        model, preprocess = clip.load("ViT-B/32", device="cpu")

    model.eval()
    # Force float32 for ONNX compatibility
    model = model.float()

    text_path = os.path.join(args.output_dir, "clip_text_encoder.onnx")
    image_path = os.path.join(args.output_dir, "clip_image_encoder.onnx")

    export_text_encoder(model, text_path)
    export_image_encoder(model, image_path)

    if args.verify:
        verify_onnx(model, preprocess, text_path, image_path)

    print(f"\n🎉 Done! ONNX models saved to {args.output_dir}")


if __name__ == "__main__":
    main()
