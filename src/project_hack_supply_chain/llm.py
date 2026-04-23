import json
import os
from typing import Any
from urllib import error, request

from dotenv import load_dotenv


load_dotenv()

DEFAULT_MODELS = {
    "claude": "claude-3-5-haiku-latest",
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4.1-mini",
    "openrouter": "openai/gpt-4.1-mini",
}

_HAS_LOGGED_LLM_WARNING = False


def _normalise_provider(raw_provider: str | None) -> str:
    provider = (raw_provider or "claude").strip().lower()
    aliases = {
        "anthropic": "claude",
        "google": "gemini",
    }
    return aliases.get(provider, provider)


def _infer_provider_from_api_key(api_key: str | None) -> str | None:
    if not api_key:
        return None
    key = api_key.strip()
    if key.startswith("AIza"):
        return "gemini"
    if key.startswith("sk-ant-"):
        return "claude"
    if key.startswith("sk-or-v1-"):
        return "openrouter"
    if key.startswith("sk-"):
        return "openai"
    return None


def get_llm_provider() -> str:
    explicit_provider = os.getenv("LLM_PROVIDER")
    if explicit_provider:
        return _normalise_provider(explicit_provider)

    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY"):
        return "claude"
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("OPENROUTER_API_KEY"):
        return "openrouter"

    inferred_provider = _infer_provider_from_api_key(os.getenv("API_KEY"))
    return inferred_provider or "claude"


def get_llm_model() -> str:
    provider = get_llm_provider()
    env_by_provider = {
        "claude": os.getenv("CLAUDE_MODEL") or os.getenv("ANTHROPIC_MODEL"),
        "gemini": os.getenv("GEMINI_MODEL"),
        "openai": os.getenv("OPENAI_MODEL"),
        "openrouter": os.getenv("OPENROUTER_MODEL"),
    }
    configured = os.getenv("LLM_MODEL") or env_by_provider.get(provider)
    return configured or DEFAULT_MODELS.get(provider, DEFAULT_MODELS["claude"])


def get_llm_api_key() -> str | None:
    provider = get_llm_provider()
    provider_keys = {
        "claude": os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY"),
        "gemini": os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        "openai": os.getenv("OPENAI_API_KEY"),
        "openrouter": os.getenv("OPENROUTER_API_KEY"),
    }
    return provider_keys.get(provider) or os.getenv("API_KEY")


def llm_is_configured() -> bool:
    provider = get_llm_provider()
    return provider in DEFAULT_MODELS and bool(get_llm_api_key())


def log_llm_configuration_warning() -> None:
    global _HAS_LOGGED_LLM_WARNING
    if _HAS_LOGGED_LLM_WARNING or llm_is_configured():
        return

    explicit_provider = os.getenv("LLM_PROVIDER")
    provider = get_llm_provider()
    message = (
        "Warning: LLM recommendations are disabled because no usable provider/API key is configured. "
        "Set LLM_PROVIDER plus a matching API key env var, or provide a compatible API_KEY fallback."
    )
    if explicit_provider:
        message += f" Current LLM_PROVIDER={explicit_provider!r}, resolved provider={provider!r}."
    else:
        message += f" No LLM_PROVIDER was set; resolved provider would be {provider!r}."

    print(message)
    _HAS_LOGGED_LLM_WARNING = True


def _post_json(url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=30) as response:
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Network error: {exc.reason}") from exc
    return json.loads(raw)


def _extract_openai_text(response_json: dict[str, Any]) -> str:
    output_text = response_json.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    output = response_json.get("output", [])
    for item in output:
        for content in item.get("content", []):
            text = content.get("text")
            if isinstance(text, str) and text.strip():
                return text
    return ""


def _extract_anthropic_text(response_json: dict[str, Any]) -> str:
    parts = response_json.get("content", [])
    texts = [part.get("text", "") for part in parts if part.get("type") == "text"]
    return "\n".join(text for text in texts if text).strip()


def _extract_gemini_text(response_json: dict[str, Any]) -> str:
    candidates = response_json.get("candidates", [])
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    texts = [part.get("text", "") for part in parts if isinstance(part.get("text"), str)]
    return "\n".join(text for text in texts if text).strip()


def _extract_openrouter_text(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices", [])
    if not choices:
        return ""
    content = choices[0].get("message", {}).get("content", "")
    return content.strip() if isinstance(content, str) else ""


def call_llm(system_prompt: str, user_prompt: str, max_output_tokens: int = 1000) -> str:
    provider = get_llm_provider()
    model = get_llm_model()
    api_key = get_llm_api_key()
    if not api_key:
        raise RuntimeError(f"No API key configured for provider '{provider}'.")

    if provider == "claude":
        response_json = _post_json(
            "https://api.anthropic.com/v1/messages",
            {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            {
                "model": model,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
                "max_tokens": max_output_tokens,
            },
        )
        return _extract_anthropic_text(response_json)

    if provider == "gemini":
        response_json = _post_json(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            {
                "x-goog-api-key": api_key,
                "content-type": "application/json",
            },
            {
                "system_instruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                "generationConfig": {
                    "temperature": 1.0,
                    "responseMimeType": "application/json",
                },
            },
        )
        return _extract_gemini_text(response_json)

    if provider == "openai":
        response_json = _post_json(
            "https://api.openai.com/v1/responses",
            {
                "Authorization": f"Bearer {api_key}",
                "content-type": "application/json",
            },
            {
                "model": model,
                "instructions": system_prompt,
                "input": user_prompt,
                "max_output_tokens": max_output_tokens,
            },
        )
        return _extract_openai_text(response_json)

    if provider == "openrouter":
        response_json = _post_json(
            "https://openrouter.ai/api/v1/chat/completions",
            {
                "Authorization": f"Bearer {api_key}",
                "content-type": "application/json",
            },
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 1.0,
                "max_tokens": max_output_tokens,
            },
        )
        return _extract_openrouter_text(response_json)

    raise RuntimeError(f"Unsupported LLM provider: {provider}")
