from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

import httpx


async def run_smoke(base_url: str, bearer_token: str, model: str) -> dict[str, Any]:
    headers = {"authorization": f"Bearer {bearer_token}"}
    request = {
        "model": model,
        "messages": [{"role": "user", "content": "Say READY in one word."}],
        "max_tokens": 8,
    }
    async with httpx.AsyncClient(timeout=180.0) as client:
        health = await client.get(f"{base_url}/health")
        completion = await client.post(f"{base_url}/v1/chat/completions", headers=headers, json=request)

        stream_chunks = 0
        async with client.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            headers=headers,
            json={**request, "stream": True, "max_tokens": 32},
        ) as stream:
            async for chunk in stream.aiter_bytes():
                if chunk:
                    stream_chunks += 1
                if stream_chunks >= 1:
                    break

    return {
        "healthStatus": health.status_code,
        "completionStatus": completion.status_code,
        "completionRunRef": completion.headers.get("x-hydralisk-run-ref"),
        "completionHasUsage": isinstance(_json_or_none(completion), dict)
        and isinstance(_json_or_none(completion).get("usage"), dict),
        "streamChunksObserved": stream_chunks,
        "passed": all(
            [
                health.status_code == 200,
                completion.status_code == 200,
                bool(completion.headers.get("x-hydralisk-run-ref")),
                stream_chunks > 0,
            ]
        ),
    }


def _json_or_none(response: httpx.Response) -> dict[str, Any] | None:
    try:
        payload = response.json()
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke a Hydralisk proxy")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--bearer-token", required=True)
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    args = parser.parse_args()
    result = asyncio.run(run_smoke(args.base_url.rstrip("/"), args.bearer_token, args.model))
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()

