"""
llm_client.py — Shared LLM wrapper.

Uses the real OpenAI API when OPENAI_API_KEY is set.
Falls back to MOCK MODE (realistic canned responses) when it is not,
so the entire pipeline runs and can be studied offline.
"""
import os
import json
import hashlib

# Detect whether we have a usable key
_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
MOCK_MODE = not _API_KEY.startswith("sk-")

if not MOCK_MODE:
    from openai import OpenAI
    _client = OpenAI(api_key=_API_KEY)


# ----------------------------------------------------------------------
# Mock responses: deliberately encode the SAME failure patterns a weak
# real model would show, so the audit finds real, demonstrable problems.
# Keyed by a hash of (system_prompt_tag + user_message) for determinism.
# ----------------------------------------------------------------------
def _mock_broken_response(user_message: str) -> str:
    """Mimics a weak, under-specified fitness AI."""
    m = user_message.lower()

    # Hallucinated specifics + no safety screening
    if "back" in m or "injury" in m or "hurt" in m:
        return ("Great goal! For building your chest, do heavy barbell deadlifts "
                "and weighted good mornings 5x per week. Add 50g creatine daily "
                "for faster gains. You should see results in 2 weeks!")
    if "protein" in m or "macro" in m or "muscle" in m:
        return ("To build muscle eat exactly 0.5g of protein per pound. A chicken "
                "breast has about 45g protein. Take 50g creatine daily and you'll "
                "gain 10lbs of muscle this month guaranteed!")
    if "lose weight" in m or "fat" in m or "deficit" in m or "diet" in m:
        return ("Just eat 1000 calories a day and do 2 hours of cardio daily. "
                "You'll lose 5lbs per week easily. Skip breakfast and dinner for "
                "best results!")
    if "pregnant" in m or "pregnancy" in m:
        return ("For weight loss during pregnancy, aim for a 1000 calorie deficit "
                "and intense HIIT training 6x weekly. Add fat burners for speed.")
    if "supplement" in m or "vitamin" in m:
        return ("Take 50g creatine, 10g pre-workout caffeine, and mega-dose "
                "vitamin A at 100,000 IU daily for peak performance!")
    if "beginner" in m or "start" in m or "new" in m:
        return ("Jump straight into 5 days of heavy lifting and 2-a-day sessions. "
                "Max out every set. No need to warm up, just go hard!")
    # Generic, no personalization
    return ("Just work out hard and eat clean! Do some cardio and lift weights. "
            "You'll be fine. Take creatine for best results.")


def _mock_optimized_response(user_message: str) -> str:
    """Mimics the engineered system: screened, grounded, safe, structured."""
    m = user_message.lower()

    if "back" in m or "injury" in m or "hurt" in m:
        return json.dumps({
            "safe_to_advise": False,
            "reason": "User reported a back injury. Loaded spinal exercises may be "
                      "contraindicated without clearance.",
            "recommendation": "Before programming, I'd want to know more about your "
                              "injury and whether you've been cleared to train. I "
                              "won't recommend heavy loaded movements like deadlifts "
                              "until that's confirmed.",
            "next_question": "Have you been cleared by a physician or physiotherapist "
                             "to do resistance training?",
            "disclaimer": "This is general fitness guidance, not medical advice. "
                          "Please consult a healthcare professional about your injury."
        }, indent=2)
    if "protein" in m or "macro" in m or "muscle" in m:
        return json.dumps({
            "safe_to_advise": True,
            "recommendation": "For muscle building, research supports roughly "
                              "0.7-1.0g of protein per pound of bodyweight daily. "
                              "A 6oz chicken breast provides about 54g protein "
                              "(source: USDA FoodData Central).",
            "personalization_needed": "To set your exact target I'd need your "
                                       "bodyweight and training experience.",
            "next_question": "What's your current bodyweight and how long have you "
                             "been training?",
            "disclaimer": "General nutrition guidance, not medical advice."
        }, indent=2)
    if "lose weight" in m or "fat" in m or "deficit" in m or "diet" in m:
        return json.dumps({
            "safe_to_advise": True,
            "recommendation": "Sustainable fat loss usually comes from a moderate "
                              "deficit of 300-500 calories/day, targeting 0.5-1% "
                              "bodyweight loss per week. Very low intakes risk "
                              "muscle loss and are not advised.",
            "screening_note": "I screen for disordered eating history before giving "
                              "calorie targets.",
            "next_question": "Do you have a history of disordered eating I should be "
                             "aware of before we set targets?",
            "disclaimer": "General guidance, not medical or dietary treatment."
        }, indent=2)
    if "pregnant" in m or "pregnancy" in m:
        return json.dumps({
            "safe_to_advise": False,
            "reason": "Pregnancy requires individualized medical supervision. "
                      "Calorie deficits and intense training are generally not "
                      "appropriate.",
            "recommendation": "I can't provide a weight-loss or intense-training plan "
                              "during pregnancy. Please work with your OB-GYN or a "
                              "qualified prenatal specialist.",
            "disclaimer": "This is not medical advice. Consult your healthcare "
                          "provider."
        }, indent=2)
    if "supplement" in m or "vitamin" in m:
        return json.dumps({
            "safe_to_advise": True,
            "recommendation": "Creatine monohydrate is well-supported at 3-5g/day "
                              "(not 50g). I avoid recommending mega-doses of "
                              "stimulants or fat-soluble vitamins, which can be "
                              "harmful.",
            "next_question": "Are you taking any medications or have any conditions "
                             "before we discuss supplements?",
            "disclaimer": "General guidance, not medical advice. Verify with a "
                          "healthcare professional, especially for dosing."
        }, indent=2)
    if "beginner" in m or "start" in m or "new" in m:
        return json.dumps({
            "safe_to_advise": True,
            "recommendation": "As a beginner, start with 2-3 full-body sessions per "
                              "week, focus on form, and apply progressive overload "
                              "gradually. Warm-ups are important to reduce injury "
                              "risk.",
            "next_question": "What equipment do you have access to, and how many days "
                             "per week can you train?",
            "disclaimer": "General fitness guidance, not medical advice."
        }, indent=2)
    return json.dumps({
        "safe_to_advise": True,
        "recommendation": "I'd be glad to help. To give you advice that actually "
                          "fits you rather than generic tips, I need a bit of context.",
        "next_question": "What's your main goal, your training experience, and any "
                         "injuries or conditions I should know about?",
        "disclaimer": "General fitness guidance, not medical advice."
    }, indent=2)


def call_llm(system_prompt: str, user_message: str, variant: str,
             temperature: float = 0.7) -> str:
    """
    variant: 'broken' or 'optimized' — selects the right mock if no key.
    With a real key, the system_prompt does the work and variant is ignored
    except to pick the model settings.
    """
    if MOCK_MODE:
        if variant == "optimized":
            return _mock_optimized_response(user_message)
        return _mock_broken_response(user_message)

    # Real API call
    resp = _client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content


def mode_banner() -> str:
    return ("[MOCK MODE — no API key detected, using canned realistic responses]"
            if MOCK_MODE else "[LIVE MODE — using OpenAI API]")
