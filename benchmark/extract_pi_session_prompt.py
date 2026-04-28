"""Extract a text-only benchmark prompt from a Pi session JSONL file.

Pi session logs can contain large image payloads and structured tool calls.
For prefill timing we mainly need the token shape of real chat turns and tool
outputs, not base64 image bytes. This script keeps text and tool-call
arguments, replaces images with compact placeholders, and prints only counts so
private prompt contents do not land in benchmark logs.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _stringify_item(item: dict[str, Any], *, include_thinking: bool) -> str | None:
    item_type = item.get("type")
    if item_type == "text":
        text = item.get("text")
        return text if isinstance(text, str) else None
    if item_type == "toolCall":
        name = item.get("name")
        args = item.get("arguments")
        if isinstance(args, str):
            return f"[tool_call {name or ''}]\n{args}"
        return f"[tool_call {name or ''}]"
    if item_type == "image":
        data = item.get("data")
        mime_type = item.get("mimeType")
        byte_count = len(data) if isinstance(data, str) else 0
        return f"[image {mime_type or 'unknown'} base64_chars={byte_count}]"
    if item_type == "thinking":
        if not include_thinking:
            return None
        thinking = item.get("thinking")
        return f"[thinking]\n{thinking}" if isinstance(thinking, str) else None
    return None


def extract_session(
    session_path: Path,
    output_path: Path,
    *,
    include_thinking: bool,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    item_counts: Counter[str] = Counter()
    lines: list[str] = []

    with session_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            counts[str(row.get("type"))] += 1
            message = row.get("message")
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "unknown")
            roles[role] += 1
            body: list[str] = []
            content = message.get("content")
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    item_counts[str(item.get("type"))] += 1
                    text = _stringify_item(item, include_thinking=include_thinking)
                    if text:
                        body.append(text)
            elif isinstance(content, str):
                body.append(content)
            if body:
                lines.append(f"\n\n<{role}>\n" + "\n\n".join(body))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prompt = "".join(lines).lstrip()
    output_path.write_text(prompt, encoding="utf-8")
    return {
        "session": str(session_path),
        "output": str(output_path),
        "message_counts": dict(counts),
        "role_counts": dict(roles),
        "item_counts": dict(item_counts),
        "output_chars": len(prompt),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-thinking",
        action="store_true",
        help="Include thinking items if they are present in the Pi log.",
    )
    args = parser.parse_args()

    summary = extract_session(
        args.session.expanduser(),
        args.output.expanduser(),
        include_thinking=bool(args.include_thinking),
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
