"""Example: OpenAI Images API."""

import os

from models.image_providers import ImageRequest, ProviderFactory

generator = ProviderFactory.create(
    provider="openai",
    model="dall-e-3",
    api_key=os.environ.get("OPENAI_API_KEY"),
)

request = ImageRequest(
    prompt="an oil painting of a fox in a library",
    style="vivid",
    width=1024,
    height=1024,
    num_images=1,
)

response = generator.generate(request)
print("Saved:", response.save("outputs/openai"))
