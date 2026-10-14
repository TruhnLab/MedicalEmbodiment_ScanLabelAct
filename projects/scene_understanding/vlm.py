"""Client for the vLLM gateway (OpenAI-compatible, LiteLLM in front), reachable from the laptop over VPN
only, not from the Slurm nodes.

Settings (landerpi/settings.py): VLLM_URL, VLLM_MODEL, VLLM_API_KEY from the environment, else from the
repo's .env (not in git, not synced to the cluster). VLLM_BLOCKED_MODELS: comma-separated parts of model
names that must never be used (user's rule; the names stay in .env, the repository is public).
"""

import base64
import json
import re
import time

import requests
import urllib3

from landerpi.settings import ENV_FILE, setting  # noqa: F401  (vlm.setting is used by the chat as well)

urllib3.disable_warnings()  # the gateway has a self-signed certificate

URL = setting("VLLM_URL")
# tested 2026-10-05 on keyframes of 20261005_124116: tight boxes incl. small items, 3-6 s per image;
# Qwen3.5-35B looped on repeated boxes, gemma-4-31B found far fewer objects
MODEL = setting("VLLM_MODEL", "nvidia/Qwen3.8-Flash-Next-NVFP4")
BLOCKED_MODELS = [b.strip().lower() for b in setting("VLLM_BLOCKED_MODELS", "").split(",") if b.strip()]


def blocked(model):
    return any(b in model.lower() for b in BLOCKED_MODELS)


def api_key():
    return setting("VLLM_API_KEY")


def image_part(jpg_bytes):
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpg_bytes).decode()}}


def chat(content, max_tokens=3000, attempts=6, timeout=300, schema=None, temperature=0):
    """One user message (text and image parts) -> answer text. No thinking: direct JSON answers.
    schema: a pydantic model the answer must follow (enforced by vLLM while generating)."""
    assert not blocked(MODEL)
    body = {"model": MODEL, "temperature": temperature, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": content}]}
    if schema is not None:
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()}}
    for i in range(attempts):
        try:
            r = requests.post(URL, json=body, headers={"Authorization": "Bearer " + api_key()},
                              verify=False, timeout=timeout)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"] or ""
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError):
            if i == attempts - 1:
                raise
            # the gateway rate-limits (HTTP 429) when other projects use it too
            time.sleep(10 * (i + 1))


def parse_json(text):
    """The first JSON object or list in an answer (models like to wrap it in ```json fences)."""
    m = re.search(r"[\[{].*[\]}]", text, re.S)
    if not m:
        raise ValueError("no JSON in answer")
    return json.loads(m[0])
