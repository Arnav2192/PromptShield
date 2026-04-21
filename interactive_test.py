from __future__ import annotations

import os
import sys
from typing import Optional

from promptshield import PromptShield


def ask(prompt: str, default: Optional[str] = None) -> str:
    if default is None:
        value = input(f"{prompt}: ").strip()
    else:
        value = input(f"{prompt} [{default}]: ").strip()
        if not value:
            value = default
    return value


def ask_multiline(prompt: str) -> str:
    print(prompt)
    print("Enter text. Finish with a single line containing only END.")
    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def fake_llm_call(prompt_text: str) -> str:
    """
    Replace this with your real LLM call.
    This placeholder just echoes the prompt so you can test the flow.
    """
    return (
        "LLM RESPONSE:\n"
        f"I received your request: {prompt_text}\n"
        "Contact me at alice@example.com if you need a follow-up."
    )


def main() -> int:
    print("=" * 72)
    print("PromptShield Interactive Test")
    print("=" * 72)

    session_id = ask("Session ID", "user-abc-123").strip()
    if not session_id:
        print("Session ID is required.")
        return 1

    tier = ask("Tier (free/premium)(f/p)", "free").strip().lower()
    if tier not in {"free", "premium", "f", "p"}:
        print("Invalid tier. Use 'free' or 'premium'.")
        return 1
    if tier == "f":
        tier = "free"
    if tier == "p":
        tier = "premium"

    ml_server_url = None
    if tier == "premium":
        ml_server_url = ask(
            "ML server URL",
            os.getenv("ML_SERVER_URL", "http://localhost:8000"),
        ).strip()
        if not ml_server_url:
            print("ML server URL is required for premium tier.")
            return 1

    redact_input = ask("Redact responses? (y/n)", "y").strip().lower()
    redact_responses = redact_input in {"y", "yes", "true", "1"}

    shield = PromptShield(
        session_id=session_id,
        tier=tier,
        ml_server_url=ml_server_url,
        redact_responses=redact_responses,
    )

    print("\n--- PromptShield initialized ---")
    print(f"session_id: {shield.session_id}")
    print(f"tier: {shield.tier}")
    print(f"redact_responses: {shield.redact_responses}")

    while True:
        print("\n" + "-" * 72)
        print("Choose an action:")
        print("  1) Inspect user prompt")
        print("  2) Inspect a RAG document")
        print("  3) Exit")

        choice = ask("Selection", "1").strip()

        if choice == "3":
            print("Exiting.")
            return 0

        if choice == "1":
            user_text = ask_multiline("Paste the user prompt")
            result = shield.inspect_input(user_text)

            print("\n--- Input inspection result ---")
            print(f"blocked: {result.blocked}")
            print(f"reason: {result.reason}")
            print(f"decoded_text: {result.decoded_text}")
            print(f"cumulative_score: {result.cumulative_score}")
            print(f"rule_result: {result.rule_result}")
            print(f"classifier_result: {result.classifier_result}")

        elif choice == "2":
            document = ask_multiline("Paste the RAG document")
            spotlighted_doc, rag_scan = shield.inspect_rag_document(document)

            print("\n--- RAG scan result ---")
            print(f"safe: {rag_scan.safe}")
            print(f"flagged_chunks: {rag_scan.flagged_chunks}")
            print(f"open_tag: {rag_scan.open_tag}")
            print(f"close_tag: {rag_scan.close_tag}")
            print("\n--- Spotlighted document ---")
            print(spotlighted_doc)

        else:
            print("Invalid selection. Choose 1, 2 OR 3.")


if __name__ == "__main__":
    raise SystemExit(main())