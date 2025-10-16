import argparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import httpx
import base64
import mimetypes
from pathlib import Path
from typing import Optional
import os
from openai import AsyncOpenAI

# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=1234)
parser.add_argument('--vllm_host', default='http://localhost:8080/v1')

args = parser.parse_args()

# Create the FastAPI application
app = FastAPI(title="vllm-proxy", version="0.1.0")

def _image_path_to_data_uri(image_path: str) -> str:
    p = Path(image_path)

    # Basic checks
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=400, detail=f"Image not found: {p}")

    # Best-effort MIME detection; default to JPEG if unknown
    mime, _ = mimetypes.guess_type(str(p))
    if not mime:
        mime = "image/jpeg"

    try:
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read/encode image: {e}")

    return f"data:{mime};base64,{b64}"

# Home route returning the app version (and the configured vllm_host for convenience)
@app.get('/')
async def version():
    return {"version": app.version, "vllm_host": args.vllm_host}

class GenerateRequest(BaseModel):
    model: Optional[str] = "qwen2_5_vl_7b"
    system_text: Optional[str] = ""
    user_text: str
    image_path: Optional[str] = ""
    temperature: float = 0.2
    max_tokens: Optional[int] = 250


def _normalize_message_content(content):
    """Normalize OpenAI/vLLM message.content to a plain string when possible."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return content

async def call_chat(payload, model: str):
    """Unified chat caller.

    - If the model name contains 'gpt', route to OpenAI Chat Completions.
    - Otherwise, route to vLLM at args.vllm_host.

    Returns the normalized text (str) when possible, else the raw content.
    Raises HTTPException on transport or API errors.
    """
    use_openai = 'gpt' in (model or '').lower()

    if use_openai:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")

        client = AsyncOpenAI(api_key=api_key)
        try:
            resp = await client.chat.completions.create(**payload)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"OpenAI request failed: {e}")

        try:
            content = resp.choices[0].message.content
        except Exception:
            raise HTTPException(status_code=500, detail="Unexpected OpenAI response shape")

        return _normalize_message_content(content)

    # vLLM path
    url = f"{args.vllm_host.rstrip('/')}/chat/completions"
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"vLLM request failed: {e}")

    data = response.json()
    if isinstance(data, dict) and "error" in data:
        # vLLM-compatible error envelope
        msg = data["error"].get("message", data["error"]) if isinstance(data["error"], dict) else data["error"]
        raise HTTPException(status_code=502, detail=f"vLLM error: {msg}")

    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        raise HTTPException(status_code=500, detail="Unexpected vLLM response shape: missing choices/message/content")

    return _normalize_message_content(content)

@app.post("/generate")
async def generate(req: GenerateRequest):

    model = req.model
    system_text = req.system_text
    user_text = req.user_text
    image_path = req.image_path
    temperature = req.temperature
    max_tokens = req.max_tokens

    # Normalize empty or whitespace-only system_text to None so we don't include an empty system message
    if system_text is not None:
        system_text = system_text.strip()
        if system_text == "":
            system_text = None

    # Normalize empty or whitespace-only image_path to None so we don't try to load an image
    if image_path is not None:
        image_path = image_path.strip()
        if image_path == "":
            image_path = None

    print(
        "Received request:\n"
        f"model='{model}',\n"
        f"system_text={system_text!r},\n"
        f"user_text={user_text!r},\n"
        f"image_path={image_path},\n"
        f"temperature={temperature},\n"
        f"max_tokens={max_tokens}"
    )

    # Build the payload, omit None values
    messages = []
    if system_text is not None:
        messages.append({"role": "system", "content": system_text})

    user_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": user_text},
        ],
    }

    # (Removed redundant no-op for max_tokens)

    if image_path is not None:
        print("Image path: '" + image_path + "'")
        data_uri = _image_path_to_data_uri(image_path)
        user_message["content"].append({
            "type": "image_url",
            "image_url": {"url": data_uri}
        })

    messages.append(user_message)

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    # Unified call: routes internally based on model name
    response_text = await call_chat(payload, model)
    print(f"Response: {response_text!r}")
    return response_text

# Launch the application with uvicorn when run as a script
if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
