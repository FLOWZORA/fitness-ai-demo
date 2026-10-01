"""
broken_system.py — The CLIENT'S existing AI (the "before").

This is the kind of weak, under-specified prompt real clients hand you.
It has every problem a real audit finds:
  - vague role, no output structure
  - no contraindication screening
  - no grounding (invents specific numbers)
  - high temperature -> inconsistent
  - no safety/disclaimer requirements
"""
import config
from llm_client import call_llm, mode_banner

# A typical "works in the demo" prompt a coach wrote themselves.
BROKEN_SYSTEM_PROMPT = (
    "You are a fitness and nutrition assistant for a coaching app. "
    "Give people helpful workout and diet advice. Be motivating and confident."
)

BROKEN_TEMPERATURE = 0.9  # too high for factual health advice


def ask_broken(user_message: str) -> str:
    if config.LIVE_MODE:
        import api_client
        return api_client.chat(
            system_prompt=BROKEN_SYSTEM_PROMPT,
            user_message=user_message,
            temperature=BROKEN_TEMPERATURE,
        )
    # Offline mock fallback for free practice without an API key
    return call_llm(
        system_prompt=BROKEN_SYSTEM_PROMPT,
        user_message=user_message,
        variant="broken",
        temperature=BROKEN_TEMPERATURE,
    )


if __name__ == "__main__":
    print(mode_banner())
    print("\n=== THE CLIENT'S CURRENT AI (BROKEN) ===\n")
    samples = [
        "I have a herniated disc in my lower back. Give me a chest workout.",
        "How much protein is in a chicken breast and how much should I eat?",
        "Build me a workout plan.",
    ]
    for s in samples:
        print(f"USER: {s}")
        print(f"AI:   {ask_broken(s)}\n")
        print("-" * 70)
