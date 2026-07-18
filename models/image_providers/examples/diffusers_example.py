"""Example: local generation with Diffusers."""

from models.image_providers import ImageRequest, ProviderFactory
from models.image_providers.config import ProviderConfig

config = ProviderConfig.from_dict({
    "provider": "diffusers",
    "model": "runwayml/stable-diffusion-v1-5",
    # "device": "cuda",            # optional, auto-detected
    # "use_xformers": True,        # optional
})

generator = ProviderFactory.create(config=config)
request = ImageRequest(
    prompt="a photorealistic mountain landscape at golden hour",
    negative_prompt="blurry, low quality",
    width=512,
    height=512,
    steps=25,
    guidance_scale=7.5,
    num_images=1,
)

response = generator.generate(request)
paths = response.save("outputs/diffusers")
print("Saved:", paths)
generator.unload()
