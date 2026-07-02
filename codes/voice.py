import pygame
import os
import hashlib
import speech_recognition as sr
from gtts import gTTS
import edge_tts
import asyncio
from openai import OpenAI
from dotenv import load_dotenv
import random
import time
import io

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY in .env")

ai_client = OpenAI(api_key=OPENAI_API_KEY)

try:
    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
except Exception as e:
    print(f"⚠️ Pygame mixer init failed: {e}")

recognizer = sr.Recognizer()
recognizer.pause_threshold = 0.8
recognizer.non_speaking_duration = 0.3
recognizer.energy_threshold = 300

mic = None
try:
    mic = sr.Microphone()
except Exception as e:
    print(f"⚠️ No microphone available, voice input disabled: {e}")

DEFAULT_LANG = "en"

if not os.path.exists("cache"):
    os.makedirs("cache")

VOICE_MAPPING = {
    "en": {"female": "en-IN-NeerjaNeural", "male": "en-IN-PrabhatNeural"},
    "hi": {"female": "hi-IN-SwaraNeural", "male": "hi-IN-MadhurNeural"},
    "te": {"female": "te-IN-ShrutiNeural", "male": "te-IN-MohanNeural"},
    "ta": {"female": "ta-IN-PallaviNeural", "male": "ta-IN-ValluvarNeural"},
    "fr": {"female": "fr-FR-DeniseNeural", "male": "fr-FR-HenriNeural"},
    "de": {"female": "de-DE-KatjaNeural", "male": "de-DE-ConradNeural"},
    "es": {"female": "es-ES-ElviraNeural", "male": "es-ES-AlvaroNeural"},
    "zh": {"female": "zh-CN-XiaoxiaoNeural", "male": "zh-CN-YunxiNeural"},
    "ja": {"female": "ja-JP-NanamiNeural", "male": "ja-JP-KeitaNeural"},
    "ar": {"female": "ar-AE-FatimaNeural", "male": "ar-AE-HamdanNeural"},
}


def get_voice_type():
    return random.choice(["male", "female"])


def speak(text, lang, voice_type, stop_requested_func):
    try:
        if voice_type is None:
            voice_type = "female"

        if isinstance(text, list):
            text = " ".join([str(x) for x in text])
        text = str(text).strip()

        if not text:
            return

        if stop_requested_func():
            return

        voice_config = VOICE_MAPPING.get(lang, VOICE_MAPPING["en"])
        selected_voice = voice_config.get(voice_type, voice_config["female"])

        voice_icon = "👩" if voice_type == "female" else "👨"
        print(f"Bot {voice_icon} ({lang}-{voice_type}): {text}")

        params_str = f"{lang}_{text}_{selected_voice}_neural"
        file_hash = hashlib.md5(params_str.encode("utf-8")).hexdigest()
        cache_path = os.path.join("cache", f"{file_hash}.mp3")

        if not os.path.exists(cache_path) or os.path.getsize(cache_path) == 0:
            try:
                async def _generate_audio():
                    communicate = edge_tts.Communicate(text, selected_voice)
                    await communicate.save(cache_path)

                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(_generate_audio())
                finally:
                    loop.close()

            except Exception as e:
                print(f"⚠️ Edge TTS failed: {e}")

            if not os.path.exists(cache_path) or os.path.getsize(cache_path) == 0:
                try:
                    tts = gTTS(text=text, lang=lang, slow=False)
                    tts.save(cache_path)
                except Exception as e_gtts:
                    print(f"❌ gTTS also failed: {e_gtts}")

        if stop_requested_func():
            return

        if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
            try:
                pygame.mixer.music.load(cache_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(30)
                    if stop_requested_func():
                        pygame.mixer.music.stop()
                        break
                try:
                    pygame.mixer.music.unload()
                except Exception:
                    pass
            except Exception as e:
                print(f"❌ Playback error: {e}")
        else:
            print("❌ Audio generation failed fully.")

    except Exception as e:
        print(f"❌ TTS error for {lang}-{voice_type}: {e}")


def listen_whisper_with_timeout(timeout_seconds=10, max_retries=2, lang="en", stop_requested_func=lambda: False):
    if mic is None:
        print("❌ No microphone available — cannot listen.")
        return "", True

    for attempt in range(max_retries + 1):
        if stop_requested_func():
            return "", True

        try:
            with mic as source:
                print("Listening...")
                audio = recognizer.listen(source, timeout=timeout_seconds, phrase_time_limit=12)

            if stop_requested_func():
                return "", True

            wav_data = audio.get_wav_data()
            audio_buffer = io.BytesIO(wav_data)
            audio_buffer.name = "audio.wav"

            transcript = ai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_buffer,
                language=lang,
            )

            text = transcript.text.strip()
            if text:
                print(f"You: {text}")
                return text, False
            else:
                if attempt < max_retries:
                    continue
                return "", True

        except (sr.WaitTimeoutError, sr.UnknownValueError):
            if attempt < max_retries:
                time.sleep(0.2)
                continue
            return "", True

        except Exception as e:
            print(f"⚠️ Whisper error: {e}")
            if attempt < max_retries:
                continue
            return "", True

    return "", True