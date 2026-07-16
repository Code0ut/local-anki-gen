from models.llm_models import LLM
from models.profile import Profile
import logging

JP_PROMPT_TEMPLATE = """
You are an expert Japanese language teacher and flashcard creator.

Your task depends on the input.

RULES

1. If the input is a Japanese WORD (contains two or more kanji/kana characters forming a vocabulary word):
- Give one natural Japanese example sentence.
- The sentence should be suitable for JLPT learners.
- The target word must appear exactly once.
- Give the reading of the sentence.
- Give the English translation.
- Write an image generation prompt describing the scene of the sentence.
- The image prompt must be in English.
- Do NOT mention text, letters, manga panels, speech bubbles, captions, or logos.

Return:

{{
    "type": "word",
    "word": "",
    "reading": "",
    "meaning": "",
    "example_sentence": "",
    "example_reading": "",
    "example_translation": "",
    "image_prompt": ""
}}

--------------------------------------------------

2. If the input is a SINGLE KANJI:
- Give all common Onyomi readings.
- Give all common Kunyomi readings.
- For each reading, provide one common vocabulary word.
- Give one natural example sentence using one of those vocabulary words.
- Give the reading of the sentence.
- Give the English translation.
- Create an English image generation prompt that visually represents the core meaning of the kanji itself.
- Do NOT include text, letters, captions, speech bubbles, or logos.

Return:

{{
    "type": "kanji",
    "kanji": "",
    "meaning": "",
    "onyomi": [
        {{
            "reading": "",
            "example_word": "",
            "meaning": ""
        }}
    ],
    "kunyomi": [
        {{
            "reading": "",
            "example_word": "",
            "meaning": ""
        }}
    ],
    "example_sentence": "",
    "example_reading": "",
    "example_translation": "",
    "image_prompt": ""
}}

--------------------------------------------------

Few-shot Examples

Input:
猫

Output:

{{
    "type":"word",
    "word":"猫",
    "reading":"ねこ",
    "meaning":"cat",
    "example_sentence":"猫が窓のそばで昼寝をしています。",
    "example_reading":"ねこ が まど の そば で ひるね を して います。",
    "example_translation":"The cat is taking a nap by the window.",
    "image_prompt":"A peaceful domestic cat sleeping beside a sunny window in a cozy Japanese home, warm afternoon light, realistic photography."
}}

--------------------------------------------------

Input:
勉強

Output:

{{
    "type":"word",
    "word":"勉強",
    "reading":"べんきょう",
    "meaning":"study",
    "example_sentence":"私は毎日日本語を勉強します。",
    "example_reading":"わたし は まいにち にほんご を べんきょう します。",
    "example_translation":"I study Japanese every day.",
    "image_prompt":"A university student studying Japanese at a desk with textbooks, notebooks, and a laptop in a bright room, realistic photography."
}}

--------------------------------------------------

Input:
水

Output:

{{
    "type":"kanji",
    "kanji":"水",
    "meaning":"water",
    "onyomi":[
        {{
            "reading":"スイ",
            "example_word":"水泳",
            "meaning":"swimming"
        }}
    ],
    "kunyomi":[
        {{
            "reading":"みず",
            "example_word":"水",
            "meaning":"water"
        }}
    ],
    "example_sentence":"冷たい水を飲みました。",
    "example_reading":"つめたい みず を のみました。",
    "example_translation":"I drank cold water.",
    "image_prompt":"Crystal clear flowing river surrounded by green forest under bright sunlight, realistic nature photography."
}}

--------------------------------------------------

Input:
山

Output:

{{
    "type":"kanji",
    "kanji":"山",
    "meaning":"mountain",
    "onyomi":[
        {{
            "reading":"サン",
            "example_word":"火山",
            "meaning":"volcano"
        }}
    ],
    "kunyomi":[
        {{
            "reading":"やま",
            "example_word":"山道",
            "meaning":"mountain trail"
        }}
    ],
    "example_sentence":"家族と山に登りました。",
    "example_reading":"かぞく と やま に のぼりました。",
    "example_translation":"I climbed a mountain with my family.",
    "image_prompt":"A tall green mountain beneath a blue sky with hiking trails and distant peaks, realistic landscape photography."
}}

--------------------------------------------------

Now process the following input.

Input:
{word}

IMPORTANT:
- Return ONLY valid JSON.
- Do not wrap the JSON in markdown.
- Do not include explanations.
- Do not include ```json.
"""

prompt = JP_PROMPT_TEMPLATE.format(word=word)

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

def sentence_gen(word,level,language,profile):
    try:
        if language not in ["en", "es", "fr", "de", "it", "pt"]:
            raise ValueError(f"Unsupported language: {language}. Supported languages are: en, es, fr, de, it, pt.")
        elif level not in ["easy", "medium", "hard"]:
            raise ValueError(f"Unsupported level: {level}. Supported levels are: easy, medium, hard.")
        elif not isinstance(profile, Profile):
            raise ValueError(f"Invalid profile object. Expected a Profile instance, got {type(profile).__name__}.")
        elif not isinstance(word, str) or not word.strip():
            raise ValueError("Invalid word. Please provide a non-empty string.")
        elif language == "jp":
            prompt = JP_PROMPT_TEMPLATE.format(word=word)
        else:
            prompt = f"Generate a sentence using the word '{word}' in {language} at a {level} difficulty level:"
        return profile.llm.generate(prompt)
    except ValueError as e:
        formatted_log = format_error_log(e, logging)
        logger.error(formatted_log)
        raise
    except Exception as e:
        formatted_log = format_error_log(e, logging)
        logger.error(formatted_log)    
        raise