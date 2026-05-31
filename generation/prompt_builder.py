from __future__ import annotations

from typing import List

from classification.intent_classifier import IntentResult
from retrieval.vector_store import RetrievedChunk


class PromptBuilder:
    def build_rag_prompt(self, query: str, retrieved_chunks: list[RetrievedChunk], intent: IntentResult) -> list[dict]:
        system_prompt = (
            "You are Nexora's AI Support Assistant. "
            "Answer only from the provided context. "
            "If the context is insufficient, say exactly: I don't have information about that. "
            f"The detected intent is {intent.intent} with confidence {intent.confidence:.2f}. "
            "Keep the tone appropriate for the intent and cite the chunk_ids you used."
        )
        context_lines = []
        for index, chunk in enumerate(retrieved_chunks, start=1):
            context_lines.append(
                f"{index}. doc_id={chunk.doc_id} chunk_id={chunk.chunk_id} content={chunk.content}"
            )
        user_prompt = (
            "Retrieved context:\n"
            + ("\n".join(context_lines) if context_lines else "No context available.")
            + f"\n\nCustomer query: {query}\n\n"
            "Provide a concise answer grounded only in the context. "
            "Explicitly mention which chunk_ids supported your answer."
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def build_clarification_prompt(self, query: str, intent: IntentResult) -> list[dict]:
        system_prompt = (
            "You are Nexora's AI Support Assistant. "
            "No relevant context was retrieved. "
            "Politely acknowledge the gap and ask one clarifying question. "
            f"The detected intent is {intent.intent} with confidence {intent.confidence:.2f}."
        )
        user_prompt = (
            f"Customer query: {query}\n\n"
            "Respond briefly, acknowledge that you do not have enough information, "
            "and ask for the minimum detail needed to proceed."
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def estimate_prompt_tokens(self, messages: list[dict]) -> int:
        return int(sum(len(message.get("content", "").split()) * 1.3 for message in messages))
