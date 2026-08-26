"""A tiny YAML-subset frontmatter parser (stdlib only).

Supports what GOAL.md / agent / skill files need: scalars, inline lists `[a, b]`, block lists
(`- item`), one level of nested mappings, comments, and quoted strings.
"""
from __future__ import annotations

import re
from typing import Any

_FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.S)


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    match = _FM_RE.match(text)
    if not match:
        return {}, text
    return parse_yaml_subset(match.group(1)), text[match.end():]


def render_frontmatter(data: dict[str, Any], body: str) -> str:
    lines = ["---"]
    for key, value in data.items():
        lines.extend(_render_item(key, value, 0))
    lines.append("---")
    return "\n".join(lines) + "\n" + body.lstrip("\n")


def _render_item(key: str, value: Any, indent: int) -> list[str]:
    pad = "  " * indent
    if isinstance(value, dict):
        out = [f"{pad}{key}:"]
        for sub_key, sub_value in value.items():
            out.extend(_render_item(sub_key, sub_value, indent + 1))
        return out
    if isinstance(value, list):
        if all(isinstance(item, (str, int, float, bool)) or item is None for item in value):
            return [f"{pad}{key}: [{', '.join(_scalar(item) for item in value)}]"]
        out = [f"{pad}{key}:"]
        for item in value:
            out.append(f"{pad}  - {_scalar(item)}")
        return out
    return [f"{pad}{key}: {_scalar(value)}"]


def _scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value)
    if text == "" or re.search(r"[:#\[\]{}',\"]|^\s|\s$", text) or text.lower() in {"true", "false", "null", "yes", "no"}:
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return text


def parse_scalar(raw: str) -> Any:
    text = raw.strip()
    if text == "" or text in {"null", "~", "None"}:
        return None
    if text.lower() in {"true", "yes"}:
        return True
    if text.lower() in {"false", "no"}:
        return False
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        inner = text[1:-1]
        return inner.replace('\\"', '"').replace("\\\\", "\\") if text[0] == '"' else inner
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(part) for part in _split_commas(inner)]
    if text.startswith("{") and text.endswith("}"):
        inner = text[1:-1].strip()
        result: dict[str, Any] = {}
        for part in _split_commas(inner):
            if ":" in part:
                key, _, value = part.partition(":")
                result[key.strip()] = parse_scalar(value)
        return result
    if re.fullmatch(r"[-+]?\d+", text):
        return int(text)
    if re.fullmatch(r"[-+]?(\d+\.\d*|\.\d+|\d+)([eE][-+]?\d+)?", text):
        return float(text)
    return text


def _split_commas(text: str) -> list[str]:
    parts, buf, depth, quote = [], [], 0, None
    for ch in text:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
            buf.append(ch)
        elif ch in "[{":
            depth += 1
            buf.append(ch)
        elif ch in "]}":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf))
    return [part.strip() for part in parts if part.strip()]


def parse_yaml_subset(text: str) -> dict[str, Any]:
    lines = []
    for raw in text.splitlines():
        stripped = raw.split(" #", 1)[0] if not raw.strip().startswith("#") else ""
        if stripped.strip():
            lines.append(stripped.rstrip())
    root: dict[str, Any] = {}
    _parse_block(lines, 0, len(lines), 0, root)
    return root


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _parse_block(lines: list[str], start: int, end: int, indent: int, target: dict[str, Any]) -> None:
    i = start
    while i < end:
        line = lines[i]
        if _indent(line) != indent:
            i += 1
            continue
        content = line.strip()
        if content.startswith("- "):
            i += 1
            continue
        key, _, rest = content.partition(":")
        key = key.strip().strip('"').strip("'")
        rest = rest.strip()
        j = i + 1
        while j < end and _indent(lines[j]) > indent:
            j += 1
        if rest:
            target[key] = parse_scalar(rest)
        elif j > i + 1 and lines[i + 1].strip().startswith("- "):
            items: list[Any] = []
            child_indent = _indent(lines[i + 1])
            k = i + 1
            while k < j:
                if _indent(lines[k]) == child_indent and lines[k].strip().startswith("- "):
                    item_text = lines[k].strip()[2:].strip()
                    if ":" in item_text and not item_text.startswith(("[", "{", '"', "'")):
                        item: dict[str, Any] = {}
                        ik, _, iv = item_text.partition(":")
                        item[ik.strip()] = parse_scalar(iv)
                        m = k + 1
                        while m < j and _indent(lines[m]) > child_indent:
                            sk, _, sv = lines[m].strip().partition(":")
                            item[sk.strip()] = parse_scalar(sv)
                            m += 1
                        items.append(item)
                        k = m
                        continue
                    items.append(parse_scalar(item_text))
                k += 1
            target[key] = items
        elif j > i + 1:
            child: dict[str, Any] = {}
            _parse_block(lines, i + 1, j, _indent(lines[i + 1]), child)
            target[key] = child
        else:
            target[key] = None
        i = j
