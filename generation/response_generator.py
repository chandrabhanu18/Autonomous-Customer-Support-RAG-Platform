from __future__ import annotations

import re
from typing import Optional

from openai import OpenAI
from pydantic import BaseModel

from config import settings
from utils import tokenize


class GeneratedResponse(BaseModel):
    response_text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ResponseGenerator:
    def __init__(self, model: str = "gpt-4o-mini", max_tokens: int = 512, temperature: float = 0.2):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client: Optional[OpenAI] = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    def _fallback_generate(self, messages: list[dict]) -> GeneratedResponse:
        user_message = messages[-1]["content"] if messages else ""
        lowered = user_message.lower()
        if "no context available" in lowered or "ask one clarifying question" in lowered:
            response_text = (
                "I don't have information about that yet. Could you share a little more detail about what you are trying to do or which part of the setup is failing?"
            )
        else:
            query_match = re.search(r"Customer query:\s*(.+)", user_message)
            query = query_match.group(1).strip() if query_match else "the request"
            chunk_pattern = re.compile(r"chunk_id=([^\s]+) content=(.+?)(?=\n\d+\. doc_id=|\Z)", re.S)
            chunks = [
                {"chunk_id": match.group(1).strip(), "content": match.group(2).strip()}
                for match in chunk_pattern.finditer(user_message)
            ]
            query_tokens = set(tokenize(query))
            ranked_chunks = sorted(
                chunks,
                key=lambda chunk: len(query_tokens & set(tokenize(chunk["content"]))),
                reverse=True,
            )
            selected = ranked_chunks[:2] if ranked_chunks else chunks[:1]
            if selected:
                summaries = []
                cited_ids = []
                for chunk in selected:
                    cited_ids.append(chunk["chunk_id"])
                    sentences = re.split(r"(?<=[.!?])\s+", chunk["content"])
                    summaries.append(" ".join(sentences[:2]).strip())
                response_text = (
                    "Based on the Nexora documentation, "
                    + " ".join(part for part in summaries if part)
                    + f" Relevant chunks: {', '.join(cited_ids)}."
                )
            else:
                response_text = "I don't have information about that yet."
        prompt_tokens = int(sum(len(message.get("content", "").split()) * 1.3 for message in messages))
        completion_tokens = max(1, int(len(response_text.split()) * 1.3))
        return GeneratedResponse(
            response_text=response_text,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def generate(self, messages: list[dict]) -> GeneratedResponse:
        if not self._client:
            return self._fallback_generate(messages)
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = response.choices[0].message.content or ""
            usage = getattr(response, "usage", None)
            prompt_tokens = int(getattr(usage, "prompt_tokens", len(messages) * 50))
            completion_tokens = int(getattr(usage, "completion_tokens", max(1, len(content.split()))))
            return GeneratedResponse(
                response_text=content.strip(),
                model=self.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
        except Exception:
            return self._fallback_generate(messages)

    def generate_with_fallback(self, messages: list[dict], fallback_messages: list[dict]) -> GeneratedResponse:
        primary = self.generate(messages)
        if len(primary.response_text) < 20 or "i don't know" in primary.response_text.lower():
            secondary = self.generate(fallback_messages)
            return secondary if len(secondary.response_text) > len(primary.response_text) else primary
        return primary
