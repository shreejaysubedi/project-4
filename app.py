"""Provenance Guard's required API. Run: flask --app app run --port 5001."""

import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.exceptions import HTTPException

from detection import combine_signals, llm_signal, stylometric_signal
from storage import AppealError, Store


SUBMISSION_LIMITS = "10 per minute;100 per day"
APPEAL_LIMITS = "10 per minute"


def create_app(test_config=None):
    load_dotenv(Path(__file__).with_name(".env"))
    app = Flask(__name__)
    app.config.from_mapping(
        DATABASE=str(Path(app.instance_path) / "provenance.sqlite3"),
        GROQ_API_KEY=os.getenv("GROQ_API_KEY", ""),
        GROQ_MODEL=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        MAX_CONTENT_LENGTH=96 * 1024,
        RATELIMIT_STORAGE_URI="memory://",
        RATELIMIT_HEADERS_ENABLED=True,
        LLM_ASSESSOR=None,
    )
    if test_config:
        app.config.update(test_config)
    store = Store(app.config["DATABASE"])
    app.extensions["store"] = store
    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=[],
        storage_uri=app.config["RATELIMIT_STORAGE_URI"],
        headers_enabled=True,
    )
    # Retain it even when disabled in tests; decorators hold a weak reference.
    app.extensions["rate_limiter"] = limiter

    def body():
        data = request.get_json()
        if not isinstance(data, dict):
            return None
        return data

    def field(data, name, minimum, maximum):
        if data is None:
            return None
        value = data.get(name)
        if not isinstance(value, str) or not minimum <= len(value.strip()) <= maximum:
            return None
        return value.strip()

    @app.errorhandler(HTTPException)
    def http_error(error):
        message = "Submission or appeal rate limit exceeded. Try again later." if error.code == 429 else error.description
        return jsonify(error={"code": error.code, "message": message}), error.code

    @app.errorhandler(AppealError)
    def appeal_error(error):
        return jsonify(error={"code": error.status_code, "message": str(error)}), error.status_code

    @app.get("/health")
    def health():
        key = app.config["GROQ_API_KEY"]
        return jsonify(
            status="ok",
            groq_configured=bool(key and key.strip() not in {"your_key_here", "your_key"}),
            model=app.config["GROQ_MODEL"],
        )

    @app.post("/submit")
    @limiter.limit(SUBMISSION_LIMITS)
    def submit():
        data = body()
        text = field(data, "text", 1, 20000)
        creator_id = field(data, "creator_id", 1, 100)
        if text is None or creator_id is None:
            return jsonify(error={
                "code": 400,
                "message": "Provide text (1–20000 characters) and creator_id (1–100 characters) as strings.",
            }), 400
        assessor = app.config["LLM_ASSESSOR"]
        llm = assessor(text) if assessor else llm_signal(
            text, app.config["GROQ_API_KEY"], app.config["GROQ_MODEL"]
        )
        result = combine_signals(llm, stylometric_signal(text))
        return jsonify(store.save_decision(text, creator_id, result))

    @app.post("/appeal")
    @limiter.limit(APPEAL_LIMITS)
    def appeal():
        data = body()
        content_id = field(data, "content_id", 1, 100)
        creator_id = field(data, "creator_id", 1, 100)
        reasoning = field(data, "creator_reasoning", 10, 4000)
        if content_id is None or creator_id is None or reasoning is None:
            return jsonify(error={
                "code": 400,
                "message": "Provide content_id, creator_id, and creator_reasoning (10–4000 characters) as strings.",
            }), 400
        return jsonify(store.appeal(content_id, creator_id, reasoning)), 201

    @app.get("/content/<content_id>")
    def content(content_id):
        record = store.get_content(content_id)
        if record is None:
            return jsonify(error={"code": 404, "message": "Content not found."}), 404
        return jsonify(record)

    @app.get("/appeals")
    def appeals():
        return jsonify(entries=store.review_queue())

    @app.get("/log")
    def log():
        try:
            limit = int(request.args.get("limit", "20"))
        except ValueError:
            limit = 0
        if not 1 <= limit <= 100:
            return jsonify(error={"code": 400, "message": "limit must be an integer from 1 to 100."}), 400
        return jsonify(entries=store.log(limit))

    return app
