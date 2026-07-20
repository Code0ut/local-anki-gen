from models.tts_providers import kokoro_provider
provider = kokoro_provider.KokoroProvider(language="ja", voice="jf_alpha")
provider.generate(
    text="こんばんわ、わたしわ　オムカル　です。",
    output_path="outputs/audio/test.wav",
    speed=1.0,
    speaker="jf_alpha",
    language="ja"
)