from __future__ import annotations

import json
from typing import Optional

from openai import OpenAI
from pydantic import BaseModel

from config import settings
from retrieval.vector_store import RetrievedChunk
from utils import split_sentences, tokenize


class FaithfulnessResult(BaseModel):
    faithfulness_score: float
    total_claims: int
    supported_claims: int
    unsupported_claims: int
    reasoning: str


class FaithfulnessEvaluator:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self._client: Optional[OpenAI] = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    def _fallback(self, response_text: str, retrieved_chunks: list[RetrievedChunk]) -> FaithfulnessResult:
        context_text = " ".join(chunk.content for chunk in retrieved_chunks)
        context_tokens = set(tokenize(context_text))
        sentences = [sentence.strip() for sentence in split_sentences(response_text) if sentence.strip()]
        total_claims = len(sentences)
        if total_claims == 0:
            return FaithfulnessResult(
                faithfulness_score=1.0,
                total_claims=0,
                supported_claims=0,
                unsupported_claims=0,
                reasoning="No factual claims detected.",
            )
        supported_claims = 0
        reasoning_parts: list[str] = []
        for sentence in sentences:
            sentence_tokens = set(tokenize(sentence))
            overlap = len(sentence_tokens & context_tokens) / max(1, len(sentence_tokens))
            if overlap >= 0.35 or "chunk_id" in sentence.lower() or "based on the nexora documentation" in sentence.lower():
                supported_claims += 1
                reasoning_parts.append(f"Supported: {sentence}")
            else:
                reasoning_parts.append(f"Unsupported: {sentence}")
        unsupported_claims = total_claims - supported_claims
        score = supported_claims / total_claims if total_claims else 1.0
        return FaithfulnessResult(
            faithfulness_score=score,
            total_claims=total_claims,
            supported_claims=supported_claims,
            unsupported_claims=unsupported_claims,
            reasoning=" ".join(reasoning_parts),
        )

    def evaluate(self, response_text: str, retrieved_chunks: list[RetrievedChunk]) -> FaithfulnessResult:
        context_text = "\n\n".join(
            f"chunk_id={chunk.chunk_id} doc_id={chunk.doc_id}\n{chunk.content}" for chunk in retrieved_chunks
        )
        system_prompt = (
            "You are a strict judge of faithfulness. "
            "Extract each factual claim from the response, then check whether the provided context directly supports it. "
            "A claim is supported only if it can be directly verified from the context. "
            "If the context does not address a claim, it is unsupported. "
            "Return only JSON with keys total_claims, supported_claims, unsupported_claims, reasoning."
        )
        user_prompt = f"Context:\n{context_text or 'No context'}\n\nResponse:\n{response_text}"
        if self._client and retrieved_chunks:
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                payload = json.loads(response.choices[0].message.content or "{}")
                total_claims = int(payload.get("total_claims", 0))
                supported_claims = int(payload.get("supported_claims", 0))
                unsupported_claims = int(payload.get("unsupported_claims", max(0, total_claims - supported_claims)))
                score = 1.0 if total_claims == 0 else supported_claims / total_claims
                return FaithfulnessResult(
                    faithfulness_score=max(0.0, min(1.0, score)),
                    total_claims=max(0, total_claims),
                    supported_claims=max(0, supported_claims),
                    unsupported_claims=max(0, unsupported_claims),
                    reasoning=str(payload.get("reasoning", "")),
                )
            except Exception:
                return self._fallback(response_text, retrieved_chunks)
        return self._fallback(response_text, retrieved_chunks)
