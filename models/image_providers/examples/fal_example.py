"""Example: Fal.ai."""

import os

from models.image_providers import ImageRequest, ProviderFactory

generator = ProviderFactory.create(
    provider="fal",
    model="fal-ai/flux/dev",
    api_key=os.environ.get("FAL_KEY"),
)

request = ImageRequest(
    prompt="a minimalist poster of a coffee cup",
    num_images=1,
)

response = generator.generate(request)
print("Saved:", response.save("outputs/fal"))
