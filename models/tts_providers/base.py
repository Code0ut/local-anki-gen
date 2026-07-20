from abc import ABC, abstractmethod
from pathlib import Path

class TTSProvider(ABC):
    @abstractmethod
    def generate_audio(self, 
                       text: str, 
                       output_path: Path,
                       speaker: str="",
                       language: str="") -> Path:
        """
        Generate audio from the given text and save it to the specified output path.

        Args:
            text (str): The input text to convert to audio.
            output_path (Path): The path where the generated audio file will be saved.
            speaker (str): The speaker to use for audio generation.
            language (str): The language to use for audio generation.
        """
        pass