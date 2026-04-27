#!/usr/bin/env python3
"""Check Markdown text diagrams for common indentation and box alignment problems.

The checker intentionally limits itself to tracked Markdown files that exist in
the worktree. It validates fenced diagram-like blocks and rejects box/tree
characters outside fences.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path


BOX_CHARS = set("┌┐└┘├┤┬┴┼─│╭╮╰╯═║╔╗╚╝╠╣╦╩╬")
TREE_CHARS = set("├└│")
ARROW_MARKERS = ("──>", "──►", "◄──", "─►", "→", "←", "=>", "-->")


@dataclass(frozen=True)
class Issue:
    path: Path
    line: int
    message: str


def display_width(text: str) -> int:
    width = 0
    for char in text:
        if char == "\t":
            width += 4
        elif unicodedata.east_asian_width(char) in {"F", "W"}:
            width += 2
        else:
            width += 1
    return width


def tracked_markdown_files(root: Path) -> list[Path]:
    output = subprocess.check_output(["git", "ls-files", "*.md"], cwd=root, text=True)
    files: list[Path] = []
    for rel in output.splitlines():
        if rel.startswith("site/"):
            continue
        path = root / rel
        if path.exists():
            files.append(path)
    return files


def fence_spans(lines: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    in_fence = False
    fence = ""
    start = 0
    for index, line in enumerate(lines, 1):
        match = re.match(r"^\s*(`{3,}|~{3,})", line)
        if match and not in_fence:
            in_fence = True
            fence = match.group(1)
            start = index
            continue
        if in_fence and re.match(r"^\s*" + re.escape(fence) + r"\s*$", line):
            spans.append((start, index))
            in_fence = False
    return spans


def fenced_blocks(lines: list[str]) -> list[tuple[int, int, list[tuple[int, str]]]]:
    blocks: list[tuple[int, int, list[tuple[int, str]]]] = []
    in_fence = False
    fence = ""
    start = 0
    body: list[tuple[int, str]] = []
    for index, line in enumerate(lines, 1):
        match = re.match(r"^\s*(`{3,}|~{3,})", line)
        if match and not in_fence:
            in_fence = True
            fence = match.group(1)
            start = index
            body = []
            continue
        if in_fence and re.match(r"^\s*" + re.escape(fence) + r"\s*$", line):
            blocks.append((start, index, body))
            in_fence = False
            continue
        if in_fence:
            body.append((index, line))
    return blocks


def is_diagram_line(line: str) -> bool:
    return any(char in line for char in BOX_CHARS) or any(marker in line for marker in ARROW_MARKERS)


def is_diagram_block(body: list[tuple[int, str]]) -> bool:
    return any(is_diagram_line(line) for _, line in body)


def line_has_tree_shape(line: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith(("├──", "└──", "│", "├─", "└─"))


def check_rectangular_box(path: Path, group: list[tuple[int, str]]) -> list[Issue]:
    issues: list[Issue] = []
    stripped = [(line_no, line.strip()) for line_no, line in group if line.strip()]
    if len(stripped) < 3:
        return issues

    indents = {len(line) - len(line.lstrip(" ")) for _, line in group if line.strip()}
    if len(indents) != 1:
        return issues

    first = stripped[0][1]
    last = stripped[-1][1]
    if sum(first.count(char) for char in "┌╔╭") > 1 or sum(last.count(char) for char in "┘╝╯") > 1:
        return issues
    if not ((first.startswith(("┌", "╔", "╭")) and first.endswith(("┐", "╗", "╮"))) and (
        last.startswith(("└", "╚", "╰")) and last.endswith(("┘", "╝", "╯"))
    )):
        return issues
    if any(any(char in line for char in "┌┐└┘╔╗╚╝╭╮╰╯") for _, line in stripped[1:-1]):
        return issues

    target_width = display_width(first)
    if display_width(last) != target_width:
        issues.append(Issue(path, stripped[-1][0], f"box bottom width {display_width(last)} != top width {target_width}"))

    for line_no, line in stripped[1:-1]:
        if not line.startswith(("│", "║")):
            continue
        border_index = max(line.rfind("│"), line.rfind("║"))
        if border_index <= 0:
            issues.append(Issue(path, line_no, "box row starts with a vertical border but does not end with one"))
            continue
        # Allow annotations after a visibly closed row, e.g. "│ field │  ← note".
        row = line[: border_index + 1]
        suffix = line[border_index + 1 :]
        if suffix and not re.match(r"^\s{2,}(?:[←#]|//|\()", suffix):
            issues.append(Issue(path, line_no, "box row has unclear text after the closing border"))
            continue
        width = display_width(row)
        if width != target_width:
            issues.append(Issue(path, line_no, f"box row width {width} != border width {target_width}"))
    return issues


def check_block(path: Path, start: int, body: list[tuple[int, str]]) -> list[Issue]:
    issues: list[Issue] = []
    if not is_diagram_block(body):
        return issues

    nonblank = [(line_no, line) for line_no, line in body if line.strip()]
    for line_no, line in nonblank:
        if "\t" in line:
            issues.append(Issue(path, line_no, "tab character inside diagram block"))
        if line.rstrip() != line:
            issues.append(Issue(path, line_no, "trailing whitespace inside diagram block"))

    # In tree diagrams, continuation rows should keep using the visible tree
    # spine instead of switching to arbitrary indentation.
    tree_lines = [(line_no, line) for line_no, line in nonblank if any(char in line for char in TREE_CHARS)]
    if tree_lines and len(tree_lines) >= 3:
        for line_no, line in tree_lines:
            if any(char in line for char in ("┌", "┐", "└", "┘")):
                continue
            if line_has_tree_shape(line):
                continue
            if re.match(r"^\s{8,}\S", line):
                issues.append(Issue(path, line_no, "tree continuation line is indented without a visible tree spine"))

    # Validate contiguous simple rectangular boxes. Complex diagrams with nested
    # boxes or arrows between boxes are left for manual review by snippets.
    current: list[tuple[int, str]] = []
    for line_no, line in body + [(body[-1][0] + 1 if body else start + 1, "")]:
        if line.strip() and any(char in line for char in BOX_CHARS):
            current.append((line_no, line))
            continue
        if current:
            issues.extend(check_rectangular_box(path, current))
            current = []

    return issues


def collect_issues(root: Path) -> tuple[list[Issue], int, int]:
    issues: list[Issue] = []
    diagram_files = 0
    diagram_blocks = 0
    for path in tracked_markdown_files(root):
        rel = path.relative_to(root)
        lines = path.read_text(errors="ignore").splitlines()
        spans = fence_spans(lines)
        has_diagram = False

        for start, _end, body in fenced_blocks(lines):
            if not is_diagram_block(body):
                continue
            has_diagram = True
            diagram_blocks += 1
            issues.extend(check_block(rel, start, body))

        def in_fence(line_no: int) -> bool:
            return any(start < line_no < end for start, end in spans)

        for line_no, line in enumerate(lines, 1):
            if in_fence(line_no):
                continue
            if any(char in line for char in BOX_CHARS) and not line.lstrip().startswith("|"):
                issues.append(Issue(rel, line_no, "box/tree character outside a fenced block"))

        if has_diagram:
            diagram_files += 1
    return issues, diagram_files, diagram_blocks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    args = parser.parse_args()

    root = args.root.resolve()
    issues, diagram_files, diagram_blocks = collect_issues(root)
    print(f"diagram_files={diagram_files}")
    print(f"diagram_blocks={diagram_blocks}")
    print(f"issues={len(issues)}")
    for issue in issues:
        print(f"{issue.path}:{issue.line}: {issue.message}")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
