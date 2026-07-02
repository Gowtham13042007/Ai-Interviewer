from __future__ import annotations
import os
import re
import json
import threading
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import SystemMessage, HumanMessage


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY in your .env file.")

QUESTIONS_STORE: dict[str, dict] = {}
SESSIONS: dict[str, "InterviewSession"] = {}


def _make_llm(temperature: float = 0.7) -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=temperature,
        api_key=OPENAI_API_KEY,
    )


_QUESTION_SYSTEM = """You are an expert technical interviewer and HR specialist.
Your task is to generate structured interview questions.
You must respond ONLY with a valid JSON array – no markdown, no extra text."""

_QUESTION_USER_TMPL = """Generate exactly {n} interview questions for the following role.

Role details
────────────
Job Title    : {job_title}
Job Type     : {job_type}
Industry     : {industry}
Experience   : {experience}
JD Excerpt   : {jd}
Key Skills   : {skills}
Focus        : {focus}
Difficulty   : {difficulty}
Tone         : {tone}
Language     : {language}

Rules
─────
1. Match difficulty and experience level precisely.
2. Mix situational / behavioral (STAR) / technical questions per the "Focus" field.
3. Write every question in the specified Language.
4. Return ONLY a JSON array of objects with these exact keys:
   - "question"   : the interview question (string)
   - "hint"       : 1-sentence answer guide for the AI interviewer (string)
   - "type"       : one of "technical" | "behavioral" | "situational" | "system-design"

Example format (truncated):
[
  {{
    "question": "Describe a time you optimised a slow database query.",
    "hint": "Look for specific metrics, tooling used, and measurable outcome.",
    "type": "behavioral"
  }}
]"""


def _parse_questions(raw: str) -> list[dict]:
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("No JSON array found in LLM response.")
    parsed = json.loads(cleaned[start: end + 1])
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON array of questions.")
    return parsed


def generate_questions_task(job_id: str, config: dict) -> None:
    QUESTIONS_STORE[job_id] = {"status": "processing", "questions": None, "parsed": None, "total": None}
    custom_questions: list[str] = config.get("custom_questions", [])
    ai_count: int = max(0, int(config.get("q_count", 8)))

    # If there's nothing for the AI to generate (e.g. all questions are custom),
    # skip the LLM call entirely rather than asking it for "exactly 0 questions".
    if ai_count == 0:
        parsed = [{"question": cq.strip(), "hint": "", "type": "custom"} for cq in custom_questions if cq.strip()]
        QUESTIONS_STORE[job_id] = {
            "status": "completed",
            "questions": None,
            "parsed": parsed,
            "total": len(parsed),
        }
        print(f"✅ Questions ready for job_id={job_id} ({len(parsed)} total, all custom)")
        return

    try:
        llm = _make_llm(temperature=0.8)
        prompt = _QUESTION_USER_TMPL.format(
            n=ai_count,
            job_title=config.get("job_title", "Not specified"),
            job_type=config.get("job_type", "Not specified"),
            industry=config.get("industry", "Not specified"),
            experience=config.get("experience", "Not specified"),
            jd=(config.get("jd") or "Not provided")[:800],
            skills=", ".join(config.get("skills", [])) or "Not specified",
            focus=config.get("focus", "mixed"),
            difficulty=config.get("difficulty", "medium"),
            tone=config.get("tone", "Professional"),
            language=config.get("language", "English"),
        )

        messages = [
            SystemMessage(content=_QUESTION_SYSTEM),
            HumanMessage(content=prompt),
        ]

        response = llm.invoke(messages)
        parsed = _parse_questions(response.content)
        for cq in custom_questions:
            if cq.strip():
                parsed.append({"question": cq.strip(), "hint": "", "type": "custom"})

        QUESTIONS_STORE[job_id] = {
            "status": "completed",
            "questions": response.content,
            "parsed": parsed,
            "total": len(parsed),
        }

        print(f"✅ Questions ready for job_id={job_id} ({len(parsed)} total)")
    except Exception as exc:
        QUESTIONS_STORE[job_id] = {"status": "failed", "error": str(exc)}
        print(f"❌ Question generation failed for job_id={job_id}: {exc}")


_INTERVIEWER_SYSTEM_TMPL = """You are a {tone} AI interviewer conducting a {difficulty}-level interview
for the role of {job_title} in the {industry} industry.

Your behaviour rules
────────────────────
• Ask ONE question at a time – never reveal upcoming questions.
• After each candidate answer: give brief, constructive spoken feedback (1-2 sentences),
  then naturally transition to the next question.
• If the candidate is off-topic or unclear, probe gently before moving on.
• Keep the conversation in {language}.
• Be encouraging but honest; match the "{tone}" tone throughout.
• Never break character. You are a real interviewer, not a chatbot.
• Do NOT say things like "As an AI…" or "I'm a language model…".

Current question list (internal – do NOT read aloud):
{questions_json}
"""

_SCORE_SYSTEM = """You are a strict but fair interview evaluator.
Given a question and a candidate's answer, respond ONLY with a JSON object:
{{"score": <integer 1-10>, "feedback": "<one sentence of specific, actionable feedback>"}}
No markdown, no extra text."""


class InterviewSession:
    def __init__(self, session_id: str, config: dict, questions: list[dict]):
        self.session_id = session_id
        self.config = config
        self.questions = questions
        self.current_index = 0
        self.history: list[dict] = []
        self.finished = False
        self.memory = InMemoryChatMessageHistory()
        self.llm = _make_llm(temperature=0.7)
        self._score_llm = _make_llm(temperature=0.0)
        self._lock = threading.Lock()
        self._system_prompt = _INTERVIEWER_SYSTEM_TMPL.format(
            tone=config.get("tone", "Professional"),
            difficulty=config.get("difficulty", "medium"),
            job_title=config.get("job_title", "the role"),
            industry=config.get("industry", "the industry"),
            language=config.get("language", "English"),
            questions_json=json.dumps(
                [{"index": i + 1, "question": q["question"]} for i, q in enumerate(questions)],
                indent=2,
            ),
        )

    def start(self) -> dict:
        first_q = self.questions[0]["question"]
        opening_prompt = (
            f"Greet the candidate warmly in a single sentence, "
            f"briefly introduce yourself as their interviewer for the {self.config.get('job_title', 'role')} position, "
            f"then ask this first question naturally (do NOT number it): {first_q}"
        )
        ai_text = self._invoke(opening_prompt)

        self.memory.add_user_message("[SESSION START]")
        self.memory.add_ai_message(ai_text)

        return {
            "session_id": self.session_id,
            "message": ai_text,
            "question": first_q,
            "hint": self.questions[0].get("hint", ""),
            "question_number": 1,
            "total": len(self.questions),
            "finished": False,
        }

    def chat(self, user_answer: str) -> dict:
        with self._lock:
            if self.finished:
                return {"message": "The interview has already ended.", "finished": True}

            q_idx = self.current_index
            q_obj = self.questions[q_idx]
            question = q_obj["question"]

            score, feedback = self._score_answer(question, user_answer, q_obj.get("hint", ""))
            self.history.append({
                "question": question,
                "answer": user_answer,
                "score": score,
                "feedback": feedback,
                "type": q_obj.get("type", "general"),
            })

            self.current_index += 1
            is_last = self.current_index >= len(self.questions)

            if is_last:
                self.finished = True
                closing = self._invoke(
                    f"The candidate just answered the final question. "
                    f"Their answer: '{user_answer}'. "
                    f"Give brief specific feedback on this answer, then wrap up the interview "
                    f"professionally in 2-3 sentences – thank them and say results will be shared."
                )
                self.memory.add_user_message(user_answer)
                self.memory.add_ai_message(closing)
                return {
                    "message": closing,
                    "score": score,
                    "feedback": feedback,
                    "question": None,
                    "hint": None,
                    "question_number": self.current_index,
                    "total": len(self.questions),
                    "finished": True,
                }

            next_q = self.questions[self.current_index]
            next_num = self.current_index + 1

            transition_prompt = (
                f"The candidate answered: '{user_answer}'. "
                f"Give them one-sentence specific feedback on that answer, "
                f"then naturally transition and ask this next question (do NOT number it or say 'next question'): "
                f"{next_q['question']}"
            )

            ai_text = self._invoke(transition_prompt)
            self.memory.add_user_message(user_answer)
            self.memory.add_ai_message(ai_text)

            return {
                "message": ai_text,
                "score": score,
                "feedback": feedback,
                "question": next_q["question"],
                "hint": next_q.get("hint", ""),
                "question_number": next_num,
                "total": len(self.questions),
                "finished": False,
            }

    def _invoke(self, user_prompt: str) -> str:
        history_msgs = self.memory.messages
        messages = [SystemMessage(content=self._system_prompt)] + history_msgs + [HumanMessage(content=user_prompt)]
        response = self.llm.invoke(messages)
        return response.content.strip()

    def _score_answer(self, question: str, answer: str, hint: str) -> tuple[int, str]:
        prompt = (
            f"Question : {question}\n"
            f"Hint     : {hint}\n"
            f"Answer   : {answer}"
        )
        messages = [
            SystemMessage(content=_SCORE_SYSTEM),
            HumanMessage(content=prompt),
        ]
        try:
            raw = self._score_llm.invoke(messages).content.strip()
            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
            data = json.loads(raw)
            score = int(data.get("score", 5))
            score = max(1, min(10, score))
            return score, str(data.get("feedback", ""))
        except Exception as exc:
            print(f"⚠️ Scoring failed: {exc}")
            return 5, "Could not evaluate this answer."


_FEEDBACK_SYSTEM = """You are a senior hiring manager writing a post-interview evaluation report.
Respond ONLY with a valid JSON object – no markdown, no extra text."""

_FEEDBACK_USER_TMPL = """Based on this completed interview session, write a comprehensive feedback report.

Role        : {job_title}
Difficulty  : {difficulty}

Full Q&A log (with per-question scores):
{qa_log}

Return a JSON object with EXACTLY these keys:
{{
  "overall_score"   : <number 1-10, one decimal allowed>,
  "recommendation"  : "Hire" | "Maybe" | "No Hire",
  "summary"         : "<2-3 sentence overall assessment>",
  "strengths"       : ["<strength 1>", "<strength 2>", "<strength 3>"],
  "improvements"    : ["<area 1>", "<area 2>", "<area 3>"],
  "question_scores" : [
    {{
      "question" : "<question text>",
      "score"    : <int 1-10>,
      "comment"  : "<one specific sentence>"
    }}
  ]
}}"""


def generate_feedback(session: InterviewSession) -> dict:
    if not session.history:
        return {
            "overall_score": 0,
            "recommendation": "No Hire",
            "summary": "No answers were recorded for this session.",
            "strengths": [],
            "improvements": [],
            "question_scores": [],
        }
    qa_log = "\n\n".join(
        f"Q{i + 1} [{item['type']}] (score {item['score']}/10):\n"
        f"  Question : {item['question']}\n"
        f"  Answer   : {item['answer']}\n"
        f"  Feedback : {item['feedback']}"
        for i, item in enumerate(session.history)
    )

    prompt = _FEEDBACK_USER_TMPL.format(
        job_title=session.config.get("job_title", "the role"),
        difficulty=session.config.get("difficulty", "medium"),
        qa_log=qa_log,
    )

    llm = _make_llm(temperature=0.3)
    messages = [
        SystemMessage(content=_FEEDBACK_SYSTEM),
        HumanMessage(content=prompt),
    ]

    response = llm.invoke(messages)
    raw = re.sub(r"```(?:json)?", "", response.content).strip().rstrip("`").strip()

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Feedback LLM did not return valid JSON.")

    return json.loads(raw[start: end + 1])