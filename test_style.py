
import torch
from diffusers import FluxPipeline
import os

# --- Configuration ---
MODEL_ID = "black-forest-labs/FLUX.1-schnell"
PROMPT = (
    "cinematic still of a young woman with short brown hair, wearing a knitted green sweater and blue jeans, "
    "standing on a train station platform at dusk, looking thoughtfully into the distance. "
    "vintage 1980s retro anime style, classic cel animation aesthetic, "
    "in the style of early Studio Ghibli and classic graphic novels, "
    "muted earthy color palette with dusty terracotta and olive green tones, "
    "delicate and clean hand-drawn ink lines, soft cel shading, matte texture, subtle film grain, "
    "a melancholic and poetic atmosphere."
)
OUTPUT_FILENAME = "flux_style_test.png"
# -------------------

def main():
    """
    Main function to run the image generation pipeline.
    """
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. This script requires a GPU to run.")
        return

    print(f"✅ CUDA available. Device: {torch.cuda.get_device_name(0)}")

    try:
        print(f"⏳ Loading FLUX model: {MODEL_ID}...")
        # Use bfloat16 for performance, compatible with 30-series cards
        pipe = FluxPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
        )
        pipe.to("cuda")
        print("✅ Model loaded successfully.")

        print(f"🎨 Generating image with prompt...")
        # FLUX.schnell is designed for few steps and no guidance
        image = pipe(
            prompt=PROMPT,
            num_inference_steps=8,
            guidance_scale=0.0,
        ).images[0]

        image.save(OUTPUT_FILENAME)
        print(f"✅ Image saved successfully to: {os.path.abspath(OUTPUT_FILENAME)}")

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        print("   Please ensure you have run 'pip install diffusers transformers accelerate torch' and are logged into Hugging Face ('huggingface-cli login').")

if __name__ == "__main__":
    main()
