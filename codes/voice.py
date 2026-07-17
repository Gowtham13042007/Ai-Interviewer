import pygame
import os
import glob
import hashlib
import logging
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
log = logging.getLogger("interviewai.voice")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY in .env")

ai_client = OpenAI(api_key=OPENAI_API_KEY)

try:
    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
except Exception as e:
    log.warning("Pygame mixer init failed: %s", e)

recognizer = sr.Recognizer()

# ─────────────────────────────────────────────────────────────────────────
# Listening tuning
# ─────────────────────────────────────────────────────────────────────────
# pause_threshold: how many seconds of silence mark "the candidate is done
# talking". 0.8s was cutting people off mid-thought (e.g. "let me think
# about that..." pauses). Bumped up so natural thinking pauses don't
# truncate the answer.
recognizer.pause_threshold = 1.2

# non_speaking_duration must be <= pause_threshold (speech_recognition
# enforces this internally); keep it a bit below the pause threshold.
recognizer.non_speaking_duration = 0.5

# energy_threshold is just a starting point now — it gets recalibrated
# against the actual room via adjust_for_ambient_noise() on every listen
# attempt below, so a stale fixed value can no longer cause "never hears
# me" (room quieter than 300) or "hears noise as speech" (room louder
# than 300) failures.
recognizer.energy_threshold = 300
recognizer.dynamic_energy_threshold = True

mic = None
try:
    mic = sr.Microphone()
except Exception as e:
    log.warning("No microphone available, voice input disabled: %s", e)

DEFAULT_LANG = "en"

CACHE_DIR = "cache"
# Cap total TTS cache size so it can't grow forever across many interviews.
CACHE_MAX_BYTES = int(os.environ.get("TTS_CACHE_MAX_MB", 200)) * 1024 * 1024

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

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


def _evict_cache_if_needed():
    """New: keeps the TTS cache folder bounded. Without this, cache/ grows
    forever since every unique (lang, text, voice) combination gets its own
    file that's never deleted."""
    try:
        files = glob.glob(os.path.join(CACHE_DIR, "*.mp3"))
        total = sum(os.path.getsize(f) for f in files)
        if total <= CACHE_MAX_BYTES:
            return
        # oldest-accessed first
        files.sort(key=lambda f: os.path.getatime(f))
        for f in files:
            if total <= CACHE_MAX_BYTES:
                break
            try:
                total -= os.path.getsize(f)
                os.remove(f)
            except OSError:
                pass
        log.info("TTS cache evicted down to ~%.1f MB", total / (1024 * 1024))
    except Exception as exc:
        log.warning("Cache eviction check failed: %s", exc)


def _run_async(coro):
    """Run an async coroutine synchronously. Prefers asyncio.run (lighter
    weight, proper cleanup) and falls back to a manual loop only if this
    thread already has a running loop bound to it."""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def speak(text, lang, voice_type, stop_requested_func, pause_requested_func=None):
    """
    pause_requested_func: optional callable; if provided and returns True,
    playback pauses (mixer paused) until it returns False again, instead of
    continuing to play through a requested pause.
    """
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

        voice_icon = "female" if voice_type == "female" else "male"
        log.info("Bot [%s/%s]: %s", lang, voice_icon, text)

        params_str = f"{lang}_{text}_{selected_voice}_neural"
        file_hash = hashlib.md5(params_str.encode("utf-8")).hexdigest()
        cache_path = os.path.join(CACHE_DIR, f"{file_hash}.mp3")

        if not os.path.exists(cache_path) or os.path.getsize(cache_path) == 0:
            try:
                async def _generate_audio():
                    communicate = edge_tts.Communicate(text, selected_voice)
                    await communicate.save(cache_path)

                _run_async(_generate_audio())

            except Exception as e:
                log.warning("Edge TTS failed: %s", e)

            if not os.path.exists(cache_path) or os.path.getsize(cache_path) == 0:
                try:
                    tts = gTTS(text=text, lang=lang, slow=False)
                    tts.save(cache_path)
                except Exception as e_gtts:
                    log.error("gTTS also failed: %s", e_gtts)

            _evict_cache_if_needed()
        else:
            # touch mtime/atime so recently-used cache entries survive eviction longer
            try:
                os.utime(cache_path, None)
            except OSError:
                pass

        if stop_requested_func():
            return

        if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
            try:
                pygame.mixer.music.load(cache_path)
                pygame.mixer.music.play()
                clock = pygame.time.Clock()  # created once, not per tick
                was_paused = False
                while pygame.mixer.music.get_busy() or was_paused:
                    clock.tick(30)
                    if stop_requested_func():
                        pygame.mixer.music.stop()
                        break
                    if pause_requested_func is not None:
                        want_pause = pause_requested_func()
                        if want_pause and not was_paused:
                            pygame.mixer.music.pause()
                            was_paused = True
                        elif not want_pause and was_paused:
                            pygame.mixer.music.unpause()
                            was_paused = False
                try:
                    pygame.mixer.music.unload()
                except Exception:
                    pass
            except Exception as e:
                log.error("Playback error: %s", e)
        else:
            log.error("Audio generation failed fully.")

    except Exception as e:
        log.error("TTS error for %s-%s: %s", lang, voice_type, e)


def listen_whisper_with_timeout(timeout_seconds=10, max_retries=2, lang="en", stop_requested_func=lambda: False):
    if mic is None:
        log.error("No microphone available — cannot listen.")
        return "", True

    for attempt in range(max_retries + 1):
        if stop_requested_func():
            return "", True

        try:
            with mic as source:
                # Recalibrate against the current room noise on every attempt
                # instead of relying on a single fixed energy_threshold set at
                # import time. This is the main fix for "it never hears me"
                # (threshold too high for a quiet room) and "it hears noise as
                # speech" (threshold too low for a noisy room) — both looked
                # identical to "the software isn't listening properly" before.
                try:
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                except Exception as calib_exc:
                    log.warning("Ambient noise calibration failed, keeping "
                                "existing threshold (%.0f): %s",
                                recognizer.energy_threshold, calib_exc)

                log.info(
                    "Listening... (attempt %d/%d, energy_threshold=%.0f)",
                    attempt + 1, max_retries + 1, recognizer.energy_threshold,
                )
                audio = recognizer.listen(
                    source,
                    timeout=timeout_seconds,
                    # 12s was truncating longer STAR-format interview answers
                    # mid-sentence. Raised so pause_threshold (silence-based
                    # end-of-speech detection) is what normally ends the
                    # phrase, with this only as a hard safety cap.
                    phrase_time_limit=45,
                )

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
                log.info("Candidate said: %s", text)
                return text, False
            else:
                log.warning(
                    "Whisper returned empty transcript (attempt %d/%d) — "
                    "audio was likely captured but contained no recognizable "
                    "speech (silence, noise, or too quiet).",
                    attempt + 1, max_retries + 1,
                )
                if attempt < max_retries:
                    continue
                return "", True

        except sr.WaitTimeoutError:
            log.warning(
                "No speech detected within %ss (attempt %d/%d) — candidate "
                "may not have started talking, or energy_threshold (%.0f) "
                "is still too high for their mic input level.",
                timeout_seconds, attempt + 1, max_retries + 1,
                recognizer.energy_threshold,
            )
            if attempt < max_retries:
                time.sleep(0.2)
                continue
            return "", True

        except sr.UnknownValueError:
            log.warning(
                "Speech captured but could not be understood by the "
                "recognizer (attempt %d/%d).",
                attempt + 1, max_retries + 1,
            )
            if attempt < max_retries:
                time.sleep(0.2)
                continue
            return "", True

        except Exception:
            # Full traceback instead of a one-line warning: silent API
            # failures (auth, rate limit, network) previously looked
            # identical to "candidate wasn't heard" from the outside.
            log.exception(
                "Whisper/mic error on attempt %d/%d",
                attempt + 1, max_retries + 1,
            )
            if attempt < max_retries:
                continue
            return "", True

    return "", True