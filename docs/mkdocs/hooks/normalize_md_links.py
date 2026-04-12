from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

HREF_RE = re.compile(r"""href=(["\'])([^"\']+)\1""")


def _rewrite_href(href: str) -> str | None:
    if not href or href.startswith("#"):
        return None
    parts = urlsplit(href)
    if parts.scheme or parts.netloc or not parts.path.endswith(".md"):
        return None
    path = parts.path
    if path.endswith("/README.md"):
        path = path[: -len("README.md")]
    elif path.endswith("/index.md"):
        path = path[: -len("index.md")]
    else:
        path = path[:-3] + "/"
    while "//" in path:
        path = path.replace("//", "/")
    if not path:
        path = "/"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))


def _rewrite_html(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        quote = match.group(1)
        href = match.group(2)
        rewritten = _rewrite_href(href)
        if not rewritten:
            return match.group(0)
        return f"href={quote}{rewritten}{quote}"

    return HREF_RE.sub(replace, text)


def on_post_build(config, **_kwargs) -> None:
    site_dir = Path(config["site_dir"])
    for html_path in site_dir.rglob("*.html"):
        original = html_path.read_text(encoding="utf-8")
        rewritten = _rewrite_html(original)
        if rewritten != original:
            html_path.write_text(rewritten, encoding="utf-8")
