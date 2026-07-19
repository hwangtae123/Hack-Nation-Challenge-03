"""Thin Flask demo for the RealDoor RAG (Stages 2-3).

Three read-only actions, all in English, none of which decides eligibility:
  * /api/understand - a cited answer to a rule question (or a safe/abstain reply)
  * /api/income     - a neutral annualized-income vs. frozen-threshold comparison
  * /api/prepare    - deterministic document-readiness flags with rule citations

Run:  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python -m rag.app
Then open http://127.0.0.1:5000
"""
from __future__ import annotations

from typing import Any

from flask import Flask, jsonify, render_template, request

from rag.calculate import IncomeSource, summarize_income
from rag.index import index_exists
from rag.prepare import assess_readiness
from rag.understand import answer_question

app = Flask(__name__)


def _income_sources(raw: list[dict[str, Any]]) -> list[IncomeSource]:
    return [
        IncomeSource(
            amount=float(s["amount"]),
            frequency=str(s["frequency"]),
            source_document_id=s.get("source_document_id"),
        )
        for s in raw
    ]


@app.get("/")
def home() -> str:
    return render_template("index.html", index_ready=index_exists())


@app.post("/api/understand")
def api_understand():
    question = (request.get_json(silent=True) or {}).get("question", "").strip()
    if not question:
        return jsonify(error="A question is required."), 400
    try:
        return jsonify(answer_question(question).to_dict())
    except FileNotFoundError:
        return jsonify(
            question=question,
            answer="The rule index has not been built yet. Run "
            "`python -m rag.build_index`, then try again.",
            abstained=True,
            safety_intercept=None,
            citations=[],
        )


@app.post("/api/income")
def api_income():
    data = request.get_json(silent=True) or {}
    try:
        sources = _income_sources(data.get("income_sources", []))
        result = summarize_income(
            sources,
            household_size=int(data.get("household_size", 1)),
            threshold_pct=int(data.get("threshold_pct", 60)),
        )
    except (KeyError, ValueError) as exc:
        return jsonify(error=f"Invalid input: {exc}"), 400
    return jsonify(result.to_dict())


@app.post("/api/prepare")
def api_prepare():
    data = request.get_json(silent=True) or {}
    try:
        assessment = assess_readiness(
            household_id=str(data.get("household_id", "HH")),
            household_size=int(data.get("household_size", 1)),
            required_document_types=list(data.get("required_document_types", [])),
            present_document_types=list(data.get("present_document_types", [])),
            income_sources=_income_sources(data.get("income_sources", [])),
            document_dates=data.get("document_dates") or None,
            known_conflicts=data.get("known_conflicts") or None,
        )
    except (KeyError, ValueError) as exc:
        return jsonify(error=f"Invalid input: {exc}"), 400
    return jsonify(assessment.to_dict())


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
