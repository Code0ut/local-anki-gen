"""Example: AUTOMATIC1111 (local WebUI API)."""

from models.image_providers import ImageRequest, ProviderFactory

generator = ProviderFactory.create(
    provider="automatic1111",
    base_url="http://127.0.0.1:7860",
    model="v1-5-pruned-emaonly.safetensors",
)

request = ImageRequest(
    prompt="studio portrait, soft lighting",
    negative_prompt="ugly, deformed",
    width=768,
    height=768,
    steps=30,
    guidance_scale=7.0,
    scheduler="DPM++ 2M Karras",
    num_images=2,
)

response = generator.generate(request)
print("Saved:", response.save("outputs/a1111"))
