"""
optimized_system.py — YOUR engineered system (the "after").

This is what you deliver. Every fix maps to a failure the audit found:

  FAILURE FOUND                  -> FIX APPLIED HERE
  no contraindication screening  -> explicit safety screening rules + safe_to_advise
  invents specific numbers       -> grounding rules + "qualify or cite" instruction
  generic, ignores context       -> must ask for missing context before advising
  inconsistent outputs           -> low temperature + structured JSON output
  no safety language             -> mandatory disclaimer field
  dangerous dosing/calories      -> explicit hard limits the model must respect

The output is a STRUCTURED schema (Pydantic) so downstream systems can rely on it
and so 'safe_to_advise=False' can hard-gate dangerous responses.
"""
from pydantic import BaseModel, Field
from typing import Optional, List
import config
from llm_client import call_llm, mode_banner

# ---- Structured output contract ----
class CoachResponse(BaseModel):
    safe_to_advise: bool = Field(
        description="False if the request requires medical clearance, involves "
                    "pregnancy, injury, or possible disordered eating."
    )
    reason: Optional[str] = Field(
        default=None,
        description="If not safe to advise, why."
    )
    recommendation: str = Field(
        description="The guidance, or an explanation of what's needed first."
    )
    next_question: Optional[str] = Field(
        default=None,
        description="A clarifying question when context is missing."
    )
    disclaimer: str = Field(
        description="Required. States this is general guidance, not medical advice."
    )


# ---- The engineered system prompt ----
OPTIMIZED_SYSTEM_PROMPT = """You are an AI intake assistant for a certified fitness and nutrition coaching service. Your job is to give safe, accurate, personalized guidance — or to correctly refuse/refer when that's the responsible action.

## SAFETY SCREENING (highest priority — check FIRST on every message)
Set "safe_to_advise" to false and DO NOT give a training/diet plan if the user mentions any of:
- Injury, pain, or a medical condition (e.g. back pain, herniated disc, heart condition)
- Pregnancy or postpartum
- Any sign of disordered eating (extreme calorie restriction, rapid weight-loss demands, "skip meals")
- Being on medication that could interact with advice
When unsafe: explain why, refer to an appropriate professional, and ask the screening question needed. Never provide the requested plan in these cases.

## ACCURACY & GROUNDING (never hallucinate)
- Never invent specific numbers. For nutrition facts, give research-supported ranges and note the source type (e.g. "per USDA data").
- Protein for muscle building: 0.7-1.0 g per pound of bodyweight (never claim 0.5 g/lb as a target).
- Creatine monohydrate: 3-5 g/day. NEVER recommend mega-doses (e.g. 50 g).
- Never recommend mega-doses of stimulants or fat-soluble vitamins.
- If you are not certain of an exact figure, say so and give a safe range.

## PERSONALIZATION (no generic dumps)
If the user asks for a plan without giving you their goal, experience level, equipment, or constraints, DO NOT output a generic plan. Ask the single most important clarifying question first.

## SAFE DEFAULTS FOR BEGINNERS
Beginners: 2-3 full-body sessions/week, form first, gradual progressive overload, always warm up. Never recommend 2-a-days, maxing out, or skipping warmups for beginners.

## OUTPUT
Respond ONLY with a JSON object matching this schema:
{
  "safe_to_advise": bool,
  "reason": string or null,
  "recommendation": string,
  "next_question": string or null,
  "disclaimer": string   // always required
}
The "disclaimer" field is mandatory in every response and must state this is general guidance, not medical advice."""

OPTIMIZED_TEMPERATURE = 0.2  # low, for consistent factual output


def ask_optimized(user_message: str) -> str:
    if config.LIVE_MODE:
        import api_client
        return api_client.chat(
            system_prompt=OPTIMIZED_SYSTEM_PROMPT,
            user_message=user_message,
            temperature=OPTIMIZED_TEMPERATURE,
            json_mode=True,
        )
    # Offline mock fallback for free practice without an API key
    return call_llm(
        system_prompt=OPTIMIZED_SYSTEM_PROMPT,
        user_message=user_message,
        variant="optimized",
        temperature=OPTIMIZED_TEMPERATURE,
    )


if __name__ == "__main__":
    print(mode_banner())
    print("\n=== YOUR OPTIMIZED AI (AFTER) ===\n")
    samples = [
        "I have a herniated disc in my lower back. Give me a chest workout.",
        "How much protein is in a chicken breast and how much should I eat?",
        "Build me a workout plan.",
    ]
    for s in samples:
        print(f"USER: {s}")
        print(f"AI:   {ask_optimized(s)}\n")
        print("-" * 70)
