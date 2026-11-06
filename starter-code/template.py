"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from typing import Any, Callable

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-flash-lite": {"input": 0.075, "output": 0.300},
    "gemini-3.5-flash-lite": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-3.5-flash-lite"
ANTHROPIC_MODEL = "claude-3-5-haiku"


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the OpenAI Chat Completions API and return the response text, latency,
    and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The OpenAI model to use (default: gpt-4o).
        temperature: Sampling temperature (0.0 – 2.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # response.usage contains input_tokens and output_tokens (prompt_tokens/completion_tokens)
    """
    from openai import OpenAI

    start = time.time()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.time() - start

    content = response.choices[0].message.content
    if isinstance(content, list):
        text = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    else:
        text = str(content)

    usage = {
        "input_tokens": getattr(response.usage, "prompt_tokens", 0),
        "output_tokens": getattr(response.usage, "completion_tokens", 0),
    }
    return text, latency, usage


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5 (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Google Gemini API (using Gemini 2.5 Flash as standard) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Gemini model to use (default: gemini-2.5-flash).
        temperature: Sampling temperature.
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        Option A (New Google GenAI SDK):
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            # Configure using types.GenerateContentConfig
            
        Option B (Legacy Google GenerativeAI SDK):
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model_inst = genai.GenerativeModel(model)
            # Configure using genai.types.GenerationConfig
            
        Ensure your usage dictionary extracts 'input_tokens' and 'output_tokens' 
        from the response metadata (e.g. response.usage_metadata).
    """
    from google import genai
    from google.genai import types

    start = time.time()
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    latency = time.time() - start

    text = str(getattr(response, "text", ""))
    usage_metadata = getattr(response, "usage_metadata", None)
    usage = {
        "input_tokens": getattr(usage_metadata, "prompt_token_count", 0),
        "output_tokens": getattr(usage_metadata, "candidates_token_count", 0),
    }
    return text, latency, usage


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Anthropic Claude API (using Claude 3.5 Haiku as default) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Claude model to use (default: claude-3-5-haiku).
        temperature: Sampling temperature (0.0 - 1.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum output tokens.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        # response.usage contains input_tokens and output_tokens
    """
    from anthropic import Anthropic

    start = time.time()
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.time() - start

    content_items = getattr(response, "content", [])
    text_parts = []
    for item in content_items:
        if hasattr(item, "text"):
            text_parts.append(str(item.text))
    text = "".join(text_parts)
    if not text and isinstance(content_items, str):
        text = content_items

    usage = {
        "input_tokens": getattr(getattr(response, "usage", None), "input_tokens", 0),
        "output_tokens": getattr(getattr(response, "usage", None), "output_tokens", 0),
    }
    return text, latency, usage


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 2.5 Flash)
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Call OpenAI (gpt-4o), OpenAI Mini (gpt-4o-mini), and Gemini 2.5 Flash (gemini-2.5-flash)
    with the same prompt and return a structured comparison dictionary.

    Calculate the exact USD token cost for input + output using the prices in PRICING_1M_TOKENS.

    Args:
        prompt: The user message to send to all models.

    Returns:
        A dictionary containing:
            - "gpt4o": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gpt4o_mini": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gemini_flash": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
    """
    gpt4o_text, gpt4o_latency, gpt4o_usage = call_openai(prompt, model=OPENAI_MODEL)
    gpt4o_mini_text, gpt4o_mini_latency, gpt4o_mini_usage = call_openai(prompt, model=OPENAI_MINI_MODEL)
    gemini_text, gemini_latency, gemini_usage = call_gemini(prompt, model=GEMINI_MODEL)

    def _compute_cost(model_name: str, usage: dict) -> float:
        pricing_key = model_name
        if model_name in {"gemini-2.5-flash-lite", "gemini-3.5-flash-lite"}:
            pricing_key = "gemini-3.5-flash-lite"
        pricing = PRICING_1M_TOKENS[pricing_key]
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        return (
            input_tokens * pricing["input"]
            + output_tokens * pricing["output"]
        ) / 1_000_000

    result = {
        "gpt4o": {
            "response": gpt4o_text,
            "latency": gpt4o_latency,
            "cost": _compute_cost("gpt-4o", gpt4o_usage),
            "input_tokens": gpt4o_usage.get("input_tokens", 0),
            "output_tokens": gpt4o_usage.get("output_tokens", 0),
        },
        "gpt4o_mini": {
            "response": gpt4o_mini_text,
            "latency": gpt4o_mini_latency,
            "cost": _compute_cost("gpt-4o-mini", gpt4o_mini_usage),
            "input_tokens": gpt4o_mini_usage.get("input_tokens", 0),
            "output_tokens": gpt4o_mini_usage.get("output_tokens", 0),
        },
        "gemini_flash": {
            "response": gemini_text,
            "latency": gemini_latency,
            "cost": _compute_cost("gemini-2.5-flash", gemini_usage),
            "input_tokens": gemini_usage.get("input_tokens", 0),
            "output_tokens": gemini_usage.get("output_tokens", 0),
        },
    }
    return result


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5 (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal using Gemini 2.5.

    Behaviour:
        - Streams response tokens from Gemini 2.5 Flash as they arrive.
        - Maintains the last 3 turns of conversation history for context.
        - Typing 'quit' or 'exit' ends the session.

    Hints:
        - Maintain a history list of conversation turns.
        - Check how to stream responses using client.chats or model.generate_content(..., stream=True).
        - Keep history limited to the last 3 turns to optimize context window and costs.
    """
    from google import genai

    history: list[str] = []
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    print("==============================================================")
    print("Vin Smart Future - Intelligent Chat Assistant")
    print("Powered by Google Gemini (gemini-3.5-flash-lite)")
    print("Type 'quit' or 'exit' to end the session.")
    print("==============================================================")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        if not user_input:
            continue

        history.append(f"User: {user_input}")
        recent_history = history[-6:]
        prompt_context = "\n".join(recent_history)

        print("Bot: ", end="", flush=True)
        stream = client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=prompt_context,
        )
        reply_parts: list[str] = []
        for chunk in stream:
            text = getattr(chunk, "text", "") or ""
            if text:
                print(text, end="", flush=True)
                reply_parts.append(text)
        print()
        history.append(f"Bot: {''.join(reply_parts)}")


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (delay = base_delay * 2^attempt).

    Args:
        fn:          Zero-argument callable to execute.
        max_retries: Maximum number of retry attempts.
        base_delay:  Initial delay in seconds before the first retry.

    Returns:
        The return value of fn() on success.

    Raises:
        The last exception raised by fn() after all retries are exhausted.
    """
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(base_delay * (2**attempt))

    if last_error is not None:
        raise last_error
    raise RuntimeError("retry_with_backoff ended without a result")


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.

    Args:
        prompts: List of prompt strings.

    Returns:
        List of dicts, each being the compare_models result with an extra
        key "prompt" containing the original prompt string.
    """
    results: list[dict] = []
    for prompt in prompts:
        try:
            item = compare_models(prompt)
        except TypeError:
            item = compare_models()
        item["prompt"] = prompt
        results.append(item)
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of batch compare results as a readable Markdown table string.

    Args:
        results: List of dicts as returned by batch_compare.

    Returns:
        A beautiful Markdown table string with columns:
        | Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |
    """
    if not results:
        return "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |\n| --- | --- | --- | --- | --- | --- |"

    def _shorten(text: str, limit: int = 50) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    def _label(model_key: str) -> str:
        labels = {
            "gpt4o": "GPT-4o",
            "gpt4o_mini": "GPT-4o-Mini",
            "gemini_flash": "Gemini-Flash",
        }
        return labels.get(model_key, model_key)

    lines = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for batch in results:
        prompt = str(batch.get("prompt", ""))
        for model_key, stats in batch.items():
            if model_key == "prompt":
                continue
            response = str(stats.get("response", ""))
            latency = float(stats.get("latency", 0.0))
            input_tokens = int(stats.get("input_tokens", 0))
            output_tokens = int(stats.get("output_tokens", 0))
            cost = float(stats.get("cost", 0.0))
            lines.append(
                f"| {prompt} | {_label(model_key)} | {_shorten(response)} | {latency:.2f}s | {input_tokens}/{output_tokens} | ${cost:.6f} |"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        # Note: Requires valid API keys set in environment variables
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini 2.5 Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")
