import os
import uuid
import logging
import threading
import time
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_cors import CORS
from dotenv import load_dotenv

from codes.ai import generate_questions_task, QUESTIONS_STORE, generate_feedback, SESSIONS, InterviewSession
from codes.voice import get_voice_type, speak, listen_whisper_with_timeout

# ─────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("interviewai.app")

LANG_CODE_MAP = {
    "english": "en",
    "hindi": "hi",
    "telugu": "te",
    "tamil": "ta",
    "french": "fr",
    "german": "de",
    "spanish": "es",
    "chinese (mandarin)": "zh",
    "japanese": "ja",
    "arabic": "ar",
    "en": "en", "hi": "hi", "te": "te", "ta": "ta",
    "fr": "fr", "de": "de", "es": "es", "zh": "zh",
    "ja": "ja", "ar": "ar",
}

# Hard caps so a malformed/abusive request can't spin up something absurd
MAX_QUESTIONS = 30
MIN_QUESTIONS = 1
JOB_WAIT_TIMEOUT_SECONDS = 45
SESSION_TTL_SECONDS = int(os.environ.get("SESSION_TTL_SECONDS", 3600))  # 1 hour
CLEANUP_INTERVAL_SECONDS = 600  # sweep every 10 min


def _lang_code(config: dict) -> str:
    raw = (config.get("language") or "en").lower()
    return LANG_CODE_MAP.get(raw, "en")


load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-in-prod")
CORS(app, supports_credentials=True)

# session_id -> live status dict, polled by the frontend
INTERVIEW_STATE: dict[str, dict] = {}
# job_id -> session_id, so refreshing /interview doesn't spawn a second voice thread
STARTED_JOBS: dict[str, str] = {}
# job_id -> threading.Event, set once question generation finishes (success or fail)
JOB_EVENTS: dict[str, threading.Event] = {}

_STATE_LOCK = threading.Lock()

# statuses that mean "this session is done, a new start is fine"
_TERMINAL_STATUSES = {"finished", "stopped", "error"}


def _new_state(total: int) -> dict:
    return {
        "status": "starting",
        "transcript": [],
        "question_number": 0,
        "total": total,
        "current_question": None,
        "hint": None,
        "finished": False,
        "error": None,
        "stop_requested": False,
        "pause_requested": False,
        "paused": False,
        "created_at": time.time(),
        "finished_at": None,
    }


def _set_state(session_id: str, **updates) -> None:
    with _STATE_LOCK:
        state = INTERVIEW_STATE.get(session_id)
        if state is None:
            return
        state.update(updates)
        if updates.get("status") in _TERMINAL_STATUSES and state.get("finished_at") is None:
            state["finished_at"] = time.time()


def _append_msg(session_id: str, role: str, text: str, score=None, feedback=None) -> None:
    with _STATE_LOCK:
        state = INTERVIEW_STATE.get(session_id)
        if state is None:
            return
        state["transcript"].append({
            "role": role, "text": text, "score": score, "feedback": feedback,
            "timestamp": time.time(),
        })


def _stop_requested_func(session_id: str):
    def _check():
        with _STATE_LOCK:
            state = INTERVIEW_STATE.get(session_id)
            return bool(state and state.get("stop_requested"))
    return _check


def _pause_requested_func(session_id: str):
    def _check():
        with _STATE_LOCK:
            state = INTERVIEW_STATE.get(session_id)
            return bool(state and state.get("pause_requested"))
    return _check


def _pause_gate(session_id: str):
    """Blocks the calling thread while pause_requested is True, without busy-spinning the CPU hard."""
    while True:
        with _STATE_LOCK:
            state = INTERVIEW_STATE.get(session_id)
            if state is None or state.get("stop_requested"):
                return
            if not state.get("pause_requested"):
                if state.get("paused"):
                    state["paused"] = False
                return
            state["paused"] = True
        time.sleep(0.3)


# ─────────────────────────────────────────────────────────────────────────
# Background session cleanup — prevents unbounded memory growth from
# INTERVIEW_STATE / SESSIONS / STARTED_JOBS / QUESTIONS_STORE / JOB_EVENTS
# ─────────────────────────────────────────────────────────────────────────
def _cleanup_loop():
    while True:
        time.sleep(CLEANUP_INTERVAL_SECONDS)
        try:
            cutoff = time.time() - SESSION_TTL_SECONDS
            with _STATE_LOCK:
                stale_sessions = [
                    sid for sid, s in INTERVIEW_STATE.items()
                    if s.get("status") in _TERMINAL_STATUSES
                    and s.get("finished_at") is not None
                    and s["finished_at"] < cutoff
                ]
                for sid in stale_sessions:
                    INTERVIEW_STATE.pop(sid, None)
                    SESSIONS.pop(sid, None)
                stale_jobs = [jid for jid, sid in STARTED_JOBS.items() if sid in stale_sessions]
                for jid in stale_jobs:
                    STARTED_JOBS.pop(jid, None)
                    QUESTIONS_STORE.pop(jid, None)
                    JOB_EVENTS.pop(jid, None)
            if stale_sessions:
                log.info("Cleaned up %d stale session(s)", len(stale_sessions))
        except Exception:
            log.exception("Cleanup loop error")


threading.Thread(target=_cleanup_loop, daemon=True).start()


def _run_interview_loop(session_id: str):
    iv_session: InterviewSession = SESSIONS[session_id]
    config = iv_session.config
    lang = _lang_code(config)
    voice_type = get_voice_type()
    stop_fn = _stop_requested_func(session_id)
    pause_fn = _pause_requested_func(session_id)

    try:
        result = iv_session.start()
        _append_msg(session_id, "ai", result["message"])
        _set_state(
            session_id,
            status="speaking",
            current_question=result.get("question"),
            hint=result.get("hint"),
            question_number=result.get("question_number", 1),
            total=result.get("total", 0),
        )
        _pause_gate(session_id)
        speak(result["message"], lang, voice_type, stop_fn, pause_fn)

        while not stop_fn():
            _pause_gate(session_id)
            if stop_fn():
                break

            _set_state(session_id, status="listening")
            answer, timed_out = listen_whisper_with_timeout(
                timeout_seconds=15, max_retries=2, lang=lang, stop_requested_func=stop_fn
            )

            if stop_fn():
                break
            if not answer:
                _append_msg(session_id, "ai", "I didn't catch that — could you repeat your answer?")
                _set_state(session_id, status="speaking")
                speak("Sorry, I didn't catch that. Could you repeat your answer?", lang, voice_type, stop_fn, pause_fn)
                continue

            _append_msg(session_id, "user", answer)
            _set_state(session_id, status="thinking")

            result = iv_session.chat(answer)

            _append_msg(session_id, "ai", result["message"], result.get("score"), result.get("feedback"))
            _set_state(
                session_id,
                current_question=result.get("question"),
                hint=result.get("hint"),
                question_number=result.get("question_number"),
                total=result.get("total", 0),
            )

            _set_state(session_id, status="speaking")
            speak(result["message"], lang, voice_type, stop_fn, pause_fn)

            if result.get("finished"):
                _set_state(session_id, status="finished", finished=True)
                return

        _set_state(session_id, status="stopped")

    except Exception as exc:
        log.exception("Interview loop error for session=%s", session_id)
        _set_state(session_id, status="error", error=str(exc))


def _generate_questions_and_signal(job_id: str, config: dict) -> None:
    try:
        generate_questions_task(job_id, config)
    finally:
        event = JOB_EVENTS.get(job_id)
        if event:
            event.set()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/interview")
def interview_page():
    config = session.get("interview_config", {})
    job_id = session.get("current_job_id")

    if not config or not job_id:
        return redirect(url_for("index"))

    return render_template("interview.html", config=config, job_id=job_id)


@app.route("/feedback")
def feedback_page():
    session_id = session.get("interview_session_id")
    if not session_id:
        return redirect(url_for("index"))
    return render_template("feedback.html", session_id=session_id)


@app.route("/health")
def health():
    with _STATE_LOCK:
        active = sum(1 for s in INTERVIEW_STATE.values() if s.get("status") not in _TERMINAL_STATUSES)
        total_sessions = len(INTERVIEW_STATE)
    return jsonify({
        "status": "ok",
        "active_sessions": active,
        "total_tracked_sessions": total_sessions,
    }), 200


def _validate_setup_payload(data: dict) -> str | None:
    """Returns an error string if invalid, else None."""
    if not (data.get("jobTitle") or "").strip():
        return "Job title is required."
    try:
        q_count = int(data.get("qCount", 8))
    except (TypeError, ValueError):
        return "Question count must be a number."
    if not (MIN_QUESTIONS <= q_count <= MAX_QUESTIONS):
        return f"Question count must be between {MIN_QUESTIONS} and {MAX_QUESTIONS}."
    return None


@app.route("/api/setup", methods=["POST"])
def save_setup():
    try:
        data = request.json or {}

        validation_error = _validate_setup_payload(data)
        if validation_error:
            return jsonify({"status": "error", "message": validation_error}), 400

        custom_questions = [q for q in data.get("customQuestions", []) if q and q.strip()]
        total_q_count = min(int(data.get("qCount", 8)), MAX_QUESTIONS)
        ai_q_count = max(0, total_q_count - len(custom_questions))

        session["interview_config"] = {
            "job_title": data.get("jobTitle"),
            "job_type": data.get("jobType"),
            "industry": data.get("industry"),
            "experience": data.get("experience"),
            "jd": data.get("jd"),
            "skills": data.get("skills", []),
            "focus": data.get("focus"),
            "language": data.get("language", "English"),
            "tone": data.get("tone"),
            "difficulty": data.get("difficulty"),
            "q_count": ai_q_count,
            # kept separately so the UI can show the real total (AI + custom)
            "total_q_count": ai_q_count + len(custom_questions),
            "custom_questions": custom_questions,
            "show_hints": bool(data.get("showHints", False)),
        }

        config = session["interview_config"]
        job_id = str(uuid.uuid4())
        session["current_job_id"] = job_id
        JOB_EVENTS[job_id] = threading.Event()

        ai_thread = threading.Thread(
            target=_generate_questions_and_signal,
            args=(job_id, config),
            daemon=True,
        )
        ai_thread.start()

        log.info("Setup saved, job_id=%s, ai_questions=%d, custom=%d", job_id, ai_q_count, len(custom_questions))

        return jsonify({
            "status": "success",
            "message": "Configuration saved, processing questions!",
            "job_id": job_id,
        }), 200

    except Exception as e:
        log.exception("save_setup failed")
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/status/<job_id>", methods=["GET"])
def get_status(job_id):
    job_data = QUESTIONS_STORE.get(job_id)
    if not job_data:
        return jsonify({"status": "not_found", "message": "No job found with this ID."}), 404
    return jsonify(job_data), 200


@app.route("/api/interview/start", methods=["POST"])
def interview_start():
    body = request.json or {}
    job_id = body.get("job_id") or session.get("current_job_id")
    config = session.get("interview_config")

    if not job_id or not config:
        return jsonify({"error": "Missing job_id or config. Please complete setup first."}), 400

    with _STATE_LOCK:
        existing_session_id = STARTED_JOBS.get(job_id)
        if existing_session_id:
            existing_state = INTERVIEW_STATE.get(existing_session_id)
            if existing_state and existing_state.get("status") not in _TERMINAL_STATUSES:
                session["interview_session_id"] = existing_session_id
                return jsonify({
                    "session_id": existing_session_id,
                    "total": existing_state.get("total", 0),
                    "resumed": True,
                }), 200

    # Event-based wait instead of a sleep-poll loop — resolves the instant
    # generation finishes rather than up to 0.4s late, and doesn't tie up
    # the worker thread spinning.
    event = JOB_EVENTS.get(job_id)
    if event is None:
        # Fallback for jobs started before this event existed (e.g. old session)
        event = threading.Event()
        job = QUESTIONS_STORE.get(job_id, {})
        if job.get("status") in ("completed", "failed"):
            event.set()

    if not event.wait(timeout=JOB_WAIT_TIMEOUT_SECONDS):
        return jsonify({"error": "Question generation timed out. Please try again."}), 504

    job = QUESTIONS_STORE.get(job_id, {})
    if job.get("status") == "failed":
        return jsonify({"error": f"Question generation failed: {job.get('error')}"}), 500

    questions = job.get("parsed", [])
    if not questions:
        return jsonify({"error": "No questions were generated. Check your configuration."}), 500

    session_id = str(uuid.uuid4())
    iv_session = InterviewSession(session_id, config, questions)
    SESSIONS[session_id] = iv_session
    session["interview_session_id"] = session_id

    with _STATE_LOCK:
        INTERVIEW_STATE[session_id] = _new_state(len(questions))
        STARTED_JOBS[job_id] = session_id

    thread = threading.Thread(target=_run_interview_loop, args=(session_id,), daemon=True)
    thread.start()

    log.info("Interview started, session_id=%s, questions=%d", session_id, len(questions))

    return jsonify({
        "session_id": session_id,
        "total": len(questions),
    }), 200


@app.route("/api/interview/status/<session_id>", methods=["GET"])
def interview_status(session_id):
    with _STATE_LOCK:
        state = INTERVIEW_STATE.get(session_id)
        if state is None:
            return jsonify({"error": "Session not found."}), 404
        return jsonify(dict(state)), 200


@app.route("/api/interview/chat", methods=["POST"])
def interview_chat():
    """
    Text-only fallback endpoint. NOT used by the current voice UI
    (interview.html drives everything through /api/interview/start +
    /api/interview/status). Only call this for a session that is NOT
    also being driven by the background voice loop — calling both at
    once will race on the same InterviewSession.
    """
    body = request.json or {}
    session_id = body.get("session_id") or session.get("interview_session_id")
    answer = (body.get("answer") or "").strip()

    if not session_id:
        return jsonify({"error": "No active session. Please start the interview first."}), 400
    if not answer:
        return jsonify({"error": "Answer cannot be empty."}), 400

    iv_session = SESSIONS.get(session_id)
    if not iv_session:
        return jsonify({"error": "Session not found. It may have expired."}), 404

    result = iv_session.chat(answer)
    return jsonify(result), 200


@app.route("/api/get-feedback", methods=["GET", "POST"])
def get_feedback():
    session_id = request.args.get("session_id") or session.get("interview_session_id")

    if not session_id:
        return jsonify({"error": "No session ID provided."}), 400

    iv_session = SESSIONS.get(session_id)

    if not iv_session:
        return jsonify({"error": "Session not found or has expired."}), 400
    if hasattr(iv_session, "_feedback_cache"):
        return jsonify(iv_session._feedback_cache), 200

    try:
        report = generate_feedback(iv_session)
        iv_session._feedback_cache = report
        return jsonify(report), 200
    except Exception as e:
        log.exception("Feedback generation error")
        return jsonify({"error": f"Could not generate feedback: {str(e)}"}), 500


@app.route("/api/interview/stop", methods=["POST"])
def interview_stop():
    body = request.json or {}
    session_id = body.get("session_id") or session.get("interview_session_id")
    if not session_id:
        return jsonify({"error": "No active session."}), 400
    _set_state(session_id, stop_requested=True, pause_requested=False)
    return jsonify({"status": "stopping"}), 200


@app.route("/api/interview/pause", methods=["POST"])
def interview_pause():
    """New: lets the candidate pause between questions (e.g. bathroom break)."""
    body = request.json or {}
    session_id = body.get("session_id") or session.get("interview_session_id")
    if not session_id:
        return jsonify({"error": "No active session."}), 400
    with _STATE_LOCK:
        state = INTERVIEW_STATE.get(session_id)
        if state is None:
            return jsonify({"error": "Session not found."}), 404
        if state.get("status") in _TERMINAL_STATUSES:
            return jsonify({"error": "Interview has already ended."}), 400
        state["pause_requested"] = True
    return jsonify({"status": "pausing"}), 200


@app.route("/api/interview/resume", methods=["POST"])
def interview_resume():
    """New: resumes a paused interview."""
    body = request.json or {}
    session_id = body.get("session_id") or session.get("interview_session_id")
    if not session_id:
        return jsonify({"error": "No active session."}), 400
    with _STATE_LOCK:
        state = INTERVIEW_STATE.get(session_id)
        if state is None:
            return jsonify({"error": "Session not found."}), 404
        state["pause_requested"] = False
        state["paused"] = False
    return jsonify({"status": "resumed"}), 200


if __name__ == "__main__":
    # threaded=True: the frontend polls /api/interview/status every second
    # while a background thread drives the voice loop, so the dev server
    # needs to service concurrent requests. For production, run behind
    # gunicorn/uwsgi instead of the built-in dev server.
    app.run(debug=os.environ.get("FLASK_DEBUG", "1") == "1", port=5000, threaded=True)