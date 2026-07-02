import os
import uuid
import threading
import time

from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_cors import CORS
from dotenv import load_dotenv

from codes.ai import generate_questions_task, QUESTIONS_STORE, generate_feedback, SESSIONS, InterviewSession
from codes.voice import get_voice_type, speak, listen_whisper_with_timeout

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
    }


def _set_state(session_id: str, **updates) -> None:
    with _STATE_LOCK:
        state = INTERVIEW_STATE.get(session_id)
        if state is None:
            return
        state.update(updates)


def _append_msg(session_id: str, role: str, text: str, score=None, feedback=None) -> None:
    with _STATE_LOCK:
        state = INTERVIEW_STATE.get(session_id)
        if state is None:
            return
        state["transcript"].append({
            "role": role, "text": text, "score": score, "feedback": feedback,
        })


def _stop_requested_func(session_id: str):
    def _check():
        with _STATE_LOCK:
            state = INTERVIEW_STATE.get(session_id)
            return bool(state and state.get("stop_requested"))
    return _check


def _run_interview_loop(session_id: str):
    iv_session: InterviewSession = SESSIONS[session_id]
    config = iv_session.config
    lang = _lang_code(config)
    voice_type = get_voice_type()
    stop_fn = _stop_requested_func(session_id)

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
        speak(result["message"], lang, voice_type, stop_fn)

        while not stop_fn():
            _set_state(session_id, status="listening")
            answer, timed_out = listen_whisper_with_timeout(
                timeout_seconds=15, max_retries=2, lang=lang, stop_requested_func=stop_fn
            )

            if stop_fn():
                break
            if not answer:
                _append_msg(session_id, "ai", "I didn't catch that — could you repeat your answer?")
                _set_state(session_id, status="speaking")
                speak("Sorry, I didn't catch that. Could you repeat your answer?", lang, voice_type, stop_fn)
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
            speak(result["message"], lang, voice_type, stop_fn)

            if result.get("finished"):
                _set_state(session_id, status="finished", finished=True)
                return

        _set_state(session_id, status="stopped")

    except Exception as exc:
        print(f"❌ Interview loop error for session={session_id}: {exc}")
        _set_state(session_id, status="error", error=str(exc))


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


@app.route("/api/setup", methods=["POST"])
def save_setup():
    try:
        data = request.json or {}
        custom_questions = [q for q in data.get("customQuestions", []) if q and q.strip()]
        total_q_count = int(data.get("qCount", 8))
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

        ai_thread = threading.Thread(
            target=generate_questions_task,
            args=(job_id, config),
            daemon=True,
        )
        ai_thread.start()

        return jsonify({
            "status": "success",
            "message": "Configuration saved, processing questions!",
            "job_id": job_id,
        }), 200

    except Exception as e:
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

    deadline = time.time() + 30
    while time.time() < deadline:
        job = QUESTIONS_STORE.get(job_id, {})
        if job.get("status") == "completed":
            break
        if job.get("status") == "failed":
            return jsonify({"error": f"Question generation failed: {job.get('error')}"}), 500
        time.sleep(0.4)
    else:
        return jsonify({"error": "Question generation timed out. Please try again."}), 504

    questions = QUESTIONS_STORE[job_id].get("parsed", [])
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
        print(f"❌ Feedback generation error: {e}")
        return jsonify({"error": f"Could not generate feedback: {str(e)}"}), 500


@app.route("/api/interview/stop", methods=["POST"])
def interview_stop():
    body = request.json or {}
    session_id = body.get("session_id") or session.get("interview_session_id")
    if not session_id:
        return jsonify({"error": "No active session."}), 400
    _set_state(session_id, stop_requested=True)
    return jsonify({"status": "stopping"}), 200


if __name__ == "__main__":
    app.run(debug=True, port=5000)