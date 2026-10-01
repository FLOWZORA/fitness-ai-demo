"""
api_client.py — Production-grade OpenAI client wrapper.

This replaces the mock-fallback llm_client.py for the LIVE version of the
project. It handles what real client work actually requires:
  - retries with backoff on transient failures
  - timeouts so a hung request doesn't freeze the pipeline
  - rate limit handling (429s)
  - running cost tracking (so you can show a client "this audit cost $0.18
    in API usage")
  - clear, actionable error messages when a key is invalid or missing

Use this for ALL real API calls. The old llm_client.py mock mode is still
used by audit.py/compare.py for free local practice — this file is what
you switch to for real client engagements and the live RAG pipeline.
"""
import time
import json
from dataclasses import dataclass, field
from openai import OpenAI, APIError, APITimeoutError, RateLimitError, AuthenticationError

import config

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not config.LIVE_MODE:
            raise RuntimeError(
                "No valid OPENAI_API_KEY found. Set it before using api_client.py:\n"
                "  PowerShell:  $env:OPENAI_API_KEY = \"sk-...\"\n"
                "  bash:        export OPENAI_API_KEY=\"sk-...\""
            )
        _client = OpenAI(api_key=config.OPENAI_API_KEY,
                         timeout=config.REQUEST_TIMEOUT_SECONDS)
    return _client


# ---------------------------------------------------------------
# Running cost tracker — so reports can show real $ spent
# ---------------------------------------------------------------
@dataclass
class UsageTracker:
    chat_input_tokens: int = 0
    chat_output_tokens: int = 0
    embedding_tokens: int = 0
    call_count: int = 0
    errors: list = field(default_factory=list)

    def add_chat(self, usage):
        self.chat_input_tokens += usage.prompt_tokens
        self.chat_output_tokens += usage.completion_tokens
        self.call_count += 1

    def add_embedding(self, usage):
        self.embedding_tokens += usage.total_tokens
        self.call_count += 1

    @property
    def estimated_cost_usd(self) -> float:
        chat_cost = (
            (self.chat_input_tokens / 1_000_000) * config.CHAT_MODEL_COST_PER_1M_INPUT
            + (self.chat_output_tokens / 1_000_000) * config.CHAT_MODEL_COST_PER_1M_OUTPUT
        )
        embed_cost = (self.embedding_tokens / 1_000_000) * config.EMBEDDING_COST_PER_1M_TOKENS
        return round(chat_cost + embed_cost, 4)

    def summary(self) -> dict:
        return {
            "api_calls": self.call_count,
            "chat_input_tokens": self.chat_input_tokens,
            "chat_output_tokens": self.chat_output_tokens,
            "embedding_tokens": self.embedding_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "errors": self.errors,
        }


usage = UsageTracker()  # shared, module-level — import this to read live cost


# ---------------------------------------------------------------
# Chat completion with retry/backoff
# ---------------------------------------------------------------
def chat(system_prompt: str, user_message: str, temperature: float = 0.7,
         model: str | None = None, json_mode: bool = False) -> str:
    """
    Calls the chat API with retries on transient errors.
    Raises a clear RuntimeError on auth failure (no point retrying that).
    """
    client = get_client()
    model = model or config.CHAT_MODEL

    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    last_error = None
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
            usage.add_chat(resp.usage)
            return resp.choices[0].message.content

        except AuthenticationError as e:
            # No point retrying a bad key
            raise RuntimeError(
                f"OpenAI authentication failed: {e}\n"
                "Check that OPENAI_API_KEY is set correctly and has not expired."
            ) from e

        except RateLimitError as e:
            wait = 2 ** attempt
            usage.errors.append(f"Rate limited (attempt {attempt}), waiting {wait}s")
            time.sleep(wait)
            last_error = e

        except APITimeoutError as e:
            usage.errors.append(f"Timeout (attempt {attempt})")
            last_error = e

        except APIError as e:
            usage.errors.append(f"API error (attempt {attempt}): {e}")
            time.sleep(1.5 ** attempt)
            last_error = e

    raise RuntimeError(
        f"OpenAI API call failed after {config.MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )


# ---------------------------------------------------------------
# Embeddings with retry/backoff
# ---------------------------------------------------------------
def embed(texts: list[str], model: str | None = None) -> list[list[float]]:
    """Embed a batch of texts. Returns one vector per input text."""
    client = get_client()
    model = model or config.EMBEDDING_MODEL

    last_error = None
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            resp = client.embeddings.create(model=model, input=texts)
            usage.add_embedding(resp.usage)
            return [item.embedding for item in resp.data]

        except AuthenticationError as e:
            raise RuntimeError(
                f"OpenAI authentication failed: {e}\n"
                "Check that OPENAI_API_KEY is set correctly."
            ) from e

        except RateLimitError as e:
            wait = 2 ** attempt
            usage.errors.append(f"Embedding rate limited, waiting {wait}s")
            time.sleep(wait)
            last_error = e

        except (APITimeoutError, APIError) as e:
            usage.errors.append(f"Embedding error (attempt {attempt}): {e}")
            time.sleep(1.5 ** attempt)
            last_error = e

    raise RuntimeError(
        f"Embedding call failed after {config.MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )


def print_usage_report():
    s = usage.summary()
    print("\n--- API Usage This Session ---")
    print(f"  Calls made:        {s['api_calls']}")
    print(f"  Chat tokens:       {s['chat_input_tokens']} in / "
          f"{s['chat_output_tokens']} out")
    print(f"  Embedding tokens:  {s['embedding_tokens']}")
    print(f"  Estimated cost:    ${s['estimated_cost_usd']}")
    if s["errors"]:
        print(f"  Recovered errors:  {len(s['errors'])} "
              f"(retried successfully)")
