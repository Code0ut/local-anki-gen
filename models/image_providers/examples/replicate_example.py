"""Example: Replicate."""

import os

from models.image_providers import ImageRequest, ProviderFactory

generator = ProviderFactory.create(
    provider="replicate",
    model="black-forest-labs/flux-schnell",
    api_key=os.environ.get("REPLICATE_API_TOKEN"),
)

request = ImageRequest(
    prompt="a watercolor illustration of a forest path",
    num_images=1,
)

response = generator.generate(request)
print("Saved:", response.save("outputs/replicate"))
