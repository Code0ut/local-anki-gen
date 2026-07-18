import json
import logging
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field
from typing import Any

from ...models.llm_providers.llm_models import LLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.FileHandler("profile.log")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
 # logging_utils.py
def format_error_log(error, trace):
    # Combine the error message with the traceback information
    log_message = "{}: {}".format(trace.format_exc() + ": " + str(error), error)
    return log_message


@dataclass
class ImageGenConfig:
    provider: str
    model: str
    api_key: str = ""
    base_url: str | None = None
    size: str = "1024x1024"
    quality: str = "high"
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "size": self.size,
            "quality": self.quality,
            **self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ImageGenConfig":
        extra = {k: v for k, v in data.items() if k not in cls.__dataclass_fields__}
        return cls(
            provider=data.get("provider", ""),
            model=data.get("model", ""),
            api_key=data.get("api_key", ""),
            base_url=data.get("base_url"),
            size=data.get("size", "1024x1024"),
            quality=data.get("quality", "high"),
            extra=extra,
        )


@dataclass
class TTSConfig:
    provider: str
    model: str
    api_key: str = ""
    base_url: str | None = None
    voice: str = "alloy"
    speed: float = 1.0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "voice": self.voice,
            "speed": self.speed,
            **self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TTSConfig":
        extra = {k: v for k, v in data.items() if k not in cls.__dataclass_fields__}
        return cls(
            provider=data.get("provider", ""),
            model=data.get("model", ""),
            api_key=data.get("api_key", ""),
            base_url=data.get("base_url"),
            voice=data.get("voice", "alloy"),
            speed=data.get("speed", 1.0),
            extra=extra,
        )


class Profile:
    SETTINGS_FILE = Path(__file__).parent /"data" / "config" / "profile_settings.json"

    def __init__(self, profile_name: str = "default"):
        self.profile_name = profile_name
        self._settings = self._load_settings()
        self._profile_data = self._settings.get("profiles", {}).get(profile_name, {})
        self.language = self._profile_data.get("language", "en")
        self._llm: LLM | None = None
        self._image_gen: ImageGenConfig | None = None
        self._tts: TTSConfig | None = None

    def _load_settings(self) -> dict:
        try:
            with open(self.SETTINGS_FILE, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.exception(f"Settings file not found: {self.SETTINGS_FILE}")
            return {}
        except json.JSONDecodeError as e:
            logger.exception(f"Invalid JSON in settings file: {e}")
            return {}

    def _save_settings(self) -> None:
        try:
            with open(self.SETTINGS_FILE, "w") as f:
                json.dump(self._settings, f, indent=2)
        except Exception as e:
            logger.exception(f"Failed to save settings: {e}")
            raise

    def _get_llm_config(self) -> dict:
        return self._profile_data.get("llm", {})

    def _get_image_gen_config(self) -> dict:
        return self._profile_data.get("image_generation", {})

    def _get_tts_config(self) -> dict:
        return self._profile_data.get("tts", {})

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            llm_config = self._get_llm_config()
            if not llm_config:
                raise ValueError(f"No LLM config found for profile '{self.profile_name}'")
            self._llm = LLM(llm_config)
        return self._llm

    @property
    def image_generation(self) -> ImageGenConfig:
        if self._image_gen is None:
            img_config = self._get_image_gen_config()
            if not img_config:
                raise ValueError(f"No image generation config found for profile '{self.profile_name}'")
            self._image_gen = ImageGenConfig.from_dict(img_config)
        return self._image_gen

    @property
    def tts(self) -> TTSConfig:
        if self._tts is None:
            tts_config = self._get_tts_config()
            if not tts_config:
                raise ValueError(f"No TTS config found for profile '{self.profile_name}'")
            self._tts = TTSConfig.from_dict(tts_config)
        return self._tts

    def get_llm_config(self) -> dict:
        return self._get_llm_config()

    def get_image_gen_config(self) -> dict:
        return self._get_image_gen_config()

    def get_tts_config(self) -> dict:
        return self._get_tts_config()

    def update_llm_config(self, config: dict) -> None:
        if "profiles" not in self._settings:
            self._settings["profiles"] = {}
        if self.profile_name not in self._settings["profiles"]:
            self._settings["profiles"][self.profile_name] = {}
        self._settings["profiles"][self.profile_name]["llm"] = config
        self._profile_data = self._settings["profiles"][self.profile_name]
        self._llm = None
        self._save_settings()

    def update_image_gen_config(self, config: dict) -> None:
        if "profiles" not in self._settings:
            self._settings["profiles"] = {}
        if self.profile_name not in self._settings["profiles"]:
            self._settings["profiles"][self.profile_name] = {}
        self._settings["profiles"][self.profile_name]["image_generation"] = config
        self._profile_data = self._settings["profiles"][self.profile_name]
        self._image_gen = None
        self._save_settings()

    def update_tts_config(self, config: dict) -> None:
        if "profiles" not in self._settings:
            self._settings["profiles"] = {}
        if self.profile_name not in self.__settings["profiles"]:
            self._settings["profiles"][self.profile_name] = {}
        self._settings["profiles"][self.profile_name]["tts"] = config
        self._profile_data = self._settings["profiles"][self.profile_name]
        self._tts = None
        self._save_settings()

    def update_llm(self, llm: LLM) -> None:
        self._llm = llm
        self.update_llm_config({
            "provider": llm.provider,
            "model": llm.model,
            "api_key": llm.api_key,
            "base_url": llm.base_url,
        })

    def update_image_gen(self, img_gen: ImageGenConfig) -> None:
        self._image_gen = img_gen
        self.update_image_gen_config(img_gen.to_dict())

    def update_tts(self, tts: TTSConfig) -> None:
        self._tts = tts
        self.update_tts_config(tts.to_dict())

    @classmethod
    def list_profiles(cls) -> list[str]:
        try:
            with open(cls.SETTINGS_FILE, "r") as f:
                settings = json.load(f)
            return list(settings.get("profiles", {}).keys())
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    @classmethod
    def create_profile(cls, profile_name: str, base_profile: str = "default") -> "Profile":
        try:
            with open(cls.SETTINGS_FILE, "r") as f:
                settings = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            settings = {"profiles": {}, "lite_llm_providers": []}

        base_data = settings.get("profiles", {}).get(base_profile, {})
        settings["profiles"][profile_name] = base_data.copy()

        with open(cls.SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)

        logger.info(f"Created profile '{profile_name}' based on '{base_profile}'")
        return cls(profile_name)

    @classmethod
    def delete_profile(cls, profile_name: str) -> bool:
        if profile_name == "default":
            logger.warning("Cannot delete default profile")
            return False

        try:
            with open(cls.SETTINGS_FILE, "r") as f:
                settings = json.load(f)

            if profile_name in settings.get("profiles", {}):
                del settings["profiles"][profile_name]
                with open(cls.SETTINGS_FILE, "w") as f:
                    json.dump(settings, f, indent=2)
                logger.info(f"Deleted profile '{profile_name}'")
                return True
            return False
        except (FileNotFoundError, json.JSONDecodeError):
            return False

    def to_dict(self) -> dict:
        return {
            "profile_name": self.profile_name,
            "llm": self.get_llm_config(),
            "image_generation": self.get_image_gen_config(),
            "tts": self.get_tts_config(),
        }

    def __repr__(self) -> str:
        return f"Profile(name={self.profile_name!r}, llm={self.get_llm_config().get('provider')}/{self.get_llm_config().get('model')})"


def get_profile(profile_name: str = "default") -> Profile:
    return Profile(profile_name)


def get_available_profiles() -> list[str]:
    return Profile.list_profiles()


def create_profile(profile_name: str, base_profile: str = "default") -> Profile:
    return Profile.create_profile(profile_name, base_profile)


def delete_profile(profile_name: str) -> bool:
    return Profile.delete_profile(profile_name)