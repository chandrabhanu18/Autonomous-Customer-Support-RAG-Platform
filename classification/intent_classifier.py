from __future__ import annotations

import json
from typing import Optional

from openai import OpenAI
from pydantic import BaseModel

from config import settings
from utils import tokenize

_ALLOWED_INTENTS = {
    "billing",
    "technical_issue",
    "feature_request",
    "integration",
    "account_management",
    "data_and_export",
    "general_inquiry",
}


class IntentResult(BaseModel):
    intent: str
    confidence: float


class IntentClassifier:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self._client: Optional[OpenAI] = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    def _heuristic_classify(self, query: str) -> IntentResult:
        text = query.lower()
        scoring = {
            "billing": ["billing", "invoice", "invoices", "pricing", "plan", "plans", "subscription", "refund", "cancel", "upgrade", "downgrade"],
            "technical_issue": ["error", "crash", "crashing", "bug", "broken", "issue", "fails", "failing", "not receiving", "problem", "troubleshoot"],
            "feature_request": ["feature", "roadmap", "request", "custom", "can i use", "would like", "add support"],
            "integration": ["integration", "integrate", "slack", "github", "jira", "connect to", "connect"],
            "account_management": ["login", "log in", "password", "two-factor", "2fa", "team member", "permissions", "permission", "invite", "account", "authenticate", "authentication"],
            "data_and_export": ["export", "backup", "csv", "data", "download", "restore"],
        }
        scores = {intent: sum(1 for keyword in keywords if keyword in text) for intent, keywords in scoring.items()}
        if "webhook" in text and ("not receiving" in text or "isn't" in text or "is not" in text):
            scores["technical_issue"] += 3
        if "forgot my password" in text or "can't log in" in text or "cannot log in" in text:
            scores["account_management"] += 4
        if all(score == 0 for score in scores.values()):
            return IntentResult(intent="general_inquiry", confidence=0.0)
        best_intent = max(scores, key=scores.get)
        total = sum(scores.values()) or 1
        confidence = min(1.0, max(0.2, scores[best_intent] / total + 0.35))
        return IntentResult(intent=best_intent, confidence=confidence)

    def classify(self, query: str) -> IntentResult:
        system_prompt = """
You are an intent classifier for Nexora support.
Return only valid JSON with keys intent and confidence.
Allowed intents:
- billing: questions about pricing, invoices, plans, refunds
- technical_issue: bugs, errors, crashes, unexpected behavior
- feature_request: asking about features, capabilities, roadmap
- integration: questions about third-party integrations
- account_management: login, permissions, team members, 2FA
- data_and_export: exports, backups, data management
- general_inquiry: anything that does not fit the above
""".strip()

        if self._client:
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                payload = json.loads(response.choices[0].message.content or "{}")
                intent = payload.get("intent", "general_inquiry")
                confidence = float(payload.get("confidence", 0.0))
                if intent not in _ALLOWED_INTENTS:
                    raise ValueError("Invalid intent")
                confidence = max(0.0, min(1.0, confidence))
                return IntentResult(intent=intent, confidence=confidence)
            except Exception:
                return self._heuristic_classify(query)
        return self._heuristic_classify(query)

    def classify_batch(self, queries: list[str]) -> list[IntentResult]:
        return [self.classify(query) for query in queries]
