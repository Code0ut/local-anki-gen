"""Example: ComfyUI (local REST API)."""

from models.image_providers import ImageRequest, ProviderFactory

generator = ProviderFactory.create(
    provider="comfyui",
    base_url="http://127.0.0.1:8188",
    model="sd_xl_base_1.0.safetensors",
)

request = ImageRequest(
    prompt="cyberpunk city skyline, neon lights",
    negative_prompt="text, watermark",
    width=1024,
    height=1024,
    steps=20,
    num_images=1,
)

response = generator.generate(request)
print("Saved:", response.save("outputs/comfyui"))
