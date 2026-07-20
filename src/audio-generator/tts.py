from models.tts_providers.base import qwen3_tts_provider
import logging
from traceback import format_exc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.FileHandler("tts.log")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
 # logging_utils.py
def format_error_log(error, trace):
    # Combine the error message with the traceback information
    log_message = "{}: {}".format(trace.format_exc() + ": " + str(error), error)
    return log_message

def generate_audio(text, output_path, speaker="", language="", instruct="Act as an friendly teacher who is bubbly and enthusiastic"):
    try:
        tts_provider = qwen3_tts_provider.Qwen3TTSProvider()
        audio_path = tts_provider.generate_audio(text, output_path, speaker, language, instruct)
        return audio_path
    except Exception as e:
        error_message = f"Error generating audio: {e}\n{format_exc()}"
        logger.error(error_message)
        raise RuntimeError(error_message) from e