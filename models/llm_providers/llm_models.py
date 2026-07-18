from litellm import completion
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.FileHandler("llm_models.log")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

class LLM:

    def __init__(self, config):
        self.provider = config["provider"]
        self.model = config["model"]
        self.api_key = config.get("api_key", "")
        self.base_url = config.get("base_url")

    def generate(
        self,
        prompt,
        system_prompt="You are a helpful assistant.",
        temperature=0.7,
        max_tokens=1000,
    ):

        kwargs = {
            "model": f"{self.provider}/{self.model}",
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if self.api_key:
            kwargs["api_key"] = self.api_key

        if self.base_url:
            kwargs["api_base"] = self.base_url

        try:
            response = completion(**kwargs)
        except Exception as e:
            logger.exception(f"Error occurred while generating LLM response: {e}")
            raise

        return response.choices[0].message.content