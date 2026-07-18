from data.config.profile import LLM, ImageGenConfig
from data.config.config import load_settings
from models.image_providers import ImageRequest, ProviderFactory
from models.image_providers.config import ProviderConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.FileHandler("sentence_generator.log")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
 # logging_utils.py
def format_error_log(error, trace):
    # Combine the error message with the traceback information
    log_message = "{}: {}".format(trace.format_exc() + ": " + str(error), error)
    return log_message
def get_image_gen_config(profile_name):
    try:
        config = load_settings(provider_type="image", json_file="data/config/profile_settings.json", profile_name=profile_name)
        return ImageGenConfig.from_dict(config)
    except Exception as e:
        logger.error(f"Error loading image generation config for profile '{profile_name}': {e}")
        raise
def generate_image(profile_name, prompt, negative_prompt=None, width=512, height=512, steps=25, guidance_scale=7.5, num_images=1,output_dir="outputs"):
    try:
        config = get_image_gen_config(profile_name)
        provider_config = ProviderConfig.from_dict({
            "provider": config.provider,
            "model": config.model,
            "api_key": config.api_key,
            "base_url": config.base_url,
            "size": config.size,
            "quality": config.quality,
            **config.extra,
        })
        generator = ProviderFactory.create(config=provider_config)
        request = ImageRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            num_images=num_images,
        )
        response = generator.generate(request)
        paths = response.save(output_dir)
        logger.info(f"Saved images: {paths}")
        generator.unload()
        return paths
    except Exception as e:
        logger.error(f"Error generating image for profile '{profile_name}': {e}")
        raise