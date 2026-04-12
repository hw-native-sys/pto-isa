# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

"""
MkDocs build-time generator for PTO Tile Lib.

We intentionally keep MkDocs config under `docs/mkdocs/` and generate a *mirror*
of repository markdown into `docs/mkdocs/src/` so the site can browse markdown
across the entire repo (README files under kernels/, tests/, scripts/, etc.).

Key property:
- Generated pages preserve original repository paths, so existing repo-relative
  links like `docs/...` or `kernels/...` keep working in the site.
"""

from __future__ import annotations

import json
import posixpath
import re
from dataclasses import dataclass
from pathlib import Path

import mkdocs_gen_files


REPO_ROOT = Path(__file__).resolve().parents[2]

SKIP_PREFIXES = (
    ".git/",
    ".github/",
    ".gitcode/",
    ".venv/",
    ".venv-mkdocs/",
    "site/",
    "site_zh/",
    "build/",
    "build_tests/",
    ".idea/",
    ".vscode/",
)

SKIP_CONTAINS = (
    "/__pycache__/",
    "/CMakeFiles/",
)

ASSET_EXTS = {
    ".svg",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bnf",
}

# Directory names whose README is the canonical index page.
# Used by _en_url_to_zh_url to map e.g. /docs/isa/ -> /docs/isa/README_zh/.
README_DIRS = {
    "coding", "isa", "machine", "assembly", "docs", "kernels",
    "tests", "demos", "scripts", "include", "cmake", "reference",
    "tutorials", "script", "package", "custom", "baseline", "add",
    "gemm_basic", "flash_atten", "gemm_performance", "a2a3", "a5",
    "kirin9030", "npu", "pto", "comm",
}


def _should_skip(rel_posix: str) -> bool:
    if rel_posix.startswith("docs/mkdocs/"):
        return True
    if rel_posix.endswith("/mkdocs.yml"):
        return True
    if rel_posix == "docs/menu_ops_development.md":
        return True
    if rel_posix.startswith(".venv"):
        return True
    if "site-packages/" in rel_posix:
        return True
    if any(rel_posix.startswith(p) for p in SKIP_PREFIXES):
        return True
    if any(s in rel_posix for s in SKIP_CONTAINS):
        return True
    if rel_posix.endswith((".pyc",)):
        return True
    return False


ABS_LINK_RE = re.compile(r'\]\(/((?!http)[^)]+)\)')
REL_IMG_RE = re.compile(r'(<img\b[^>]*\bsrc=["\'])((?!http|/|data:)[^"\'>]+)(["\'])')


def _rewrite_rel_imgs_for_build(text: str, src_rel: str) -> str:
    """Rewrite relative <img src="..."> paths so they resolve correctly from
    the MkDocs virtual page URL.

    MkDocs serves foo/bar.md at /foo/bar/, so a relative image path that
    works when browsing the repo (relative to foo/) needs to be adjusted
    to be relative to /foo/bar/ instead.

    Example:
      src_rel = "docs/getting-started.md"
      img_path = "figures/pto_logo.svg"  (relative to docs/)
      resolved repo path = docs/figures/pto_logo.svg
      MkDocs page URL   = /docs/getting-started/
      correct rel path  = ../figures/pto_logo.svg
    """
    # Directory containing the source file (repo-relative, posix).
    src_dir = Path(src_rel).parent.as_posix()  # e.g. "docs"

    # Virtual page directory (where MkDocs serves the page).
    # For foo/bar.md -> /foo/bar/  so page_dir = "foo/bar"
    page_dir = Path(src_rel).with_suffix('').as_posix()  # e.g. "docs/getting-started"

    def replace(m: re.Match) -> str:
        prefix, img_path, suffix = m.group(1), m.group(2), m.group(3)
        # Resolve image to repo-relative path.
        repo_img = (src_dir + '/' + img_path) if (src_dir and src_dir != '.') else img_path
        # Normalize (handle any ../ in original img_path).
        repo_img = posixpath.normpath(repo_img)
        # Compute relative path from page_dir to repo_img.
        rel = posixpath.relpath(repo_img, page_dir)  # e.g. ../figures/pto_logo.svg
        return f'{prefix}{rel}{suffix}'

    return REL_IMG_RE.sub(replace, text)


# Matches a relative markdown link that starts with one or more "../" components.
# Group 1: the "../" prefix (one or more), Group 2: the rest of the target.
REL_LINK_RE = re.compile(r'\]\((\.\./+)([^)]+)\)')

# Matches repo-relative links into docs/mkdocs/src/ (e.g. mkdocs/src/manual/foo.md).
# These appear in repo-level docs so they resolve correctly during static browsing,
# but must be rewritten to root-absolute paths at MkDocs build time.
MKDOCS_SRC_LINK_RE = re.compile(r'\]\(mkdocs/src/([^)]+)\)')

# Strip stale "<!-- Generated from ... -->" header lines that accumulate on
# repeated builds when docs_dir is the same directory as the source files.
_GENERATED_HEADER_RE = re.compile(
    r'^(?:<!-- Generated from `[^`]*` -->\s*\n\n?)+', re.MULTILINE
)


def _strip_generated_header(text: str) -> str:
    """Remove any leading '<!-- Generated from ... -->' comment blocks."""
    return _GENERATED_HEADER_RE.sub('', text)


def _rewrite_links_for_build(text: str, virtual_path: str) -> str:
    """Rewrite links in a hand-written docs/mkdocs/src/ file so they resolve
    correctly from the MkDocs virtual page URL.

    Two kinds of links are rewritten:

    1. Root-absolute links like /docs/isa/TADD.md  ->  ../docs/isa/TADD.md
       These are written with a leading '/' so they work when browsing the
       repo on GitHub/Gitee; at build time they need to be relative.

    2. Relative links whose "../" depth is wrong for the virtual path.
       Example: hand-written file lives at
         docs/mkdocs/src/manual/appendix-d.md  (repo path)
       so the author wrote  ../../docs/isa/TADD.md  (2 levels up from
       docs/mkdocs/src/manual/ to reach the repo root docs/).
       But the virtual path is  manual/appendix-d.md  (depth=1), so
       MkDocs needs  ../docs/isa/TADD.md  (only 1 level up).

       The hand-written source sits at depth
         src_depth = len(Path("docs/mkdocs/src") / virtual_path).parent.parts
                   = len(("docs","mkdocs","src","manual")) = 4
       and the virtual path sits at depth
         virt_depth = len(Path(virtual_path).parent.parts)
                    = len(("manual",)) = 1
       so each "../" in the original link corresponds to climbing one level
       in the repo tree.  After stripping the src prefix the correct number
       of "../" is virt_depth.

    Args:
        text:         Markdown source text.
        virtual_path: Virtual path of the file (relative to docs_dir), e.g.
                      "manual/appendix-d-instruction-family-matrix.md".
    """
    # Depth of virtual page's parent directory.
    virt_depth = len(Path(virtual_path).parent.parts)  # e.g. 1 for "manual/foo.md"

    # Depth of the source file inside docs/mkdocs/src/.
    src_depth = len((Path("docs") / "mkdocs" / "src" / virtual_path).parent.parts)

    # --- Pass 1: root-absolute links /foo/bar  ->  (../)*virt_depth foo/bar ---
    prefix_abs = '../' * virt_depth if virt_depth else ''

    def replace_abs(m: re.Match) -> str:
        return f']({prefix_abs}{m.group(1)})'

    text = ABS_LINK_RE.sub(replace_abs, text)

    # --- Pass 2: relative links with wrong ../ depth ---
    # The author wrote the link relative to the *repo* source file location
    # (src_depth levels deep).  We need it relative to the virtual page
    # (virt_depth levels deep).  We only touch links whose leading "../"
    # count equals src_depth (exactly what the author would write to reach
    # the repo root from the source file).
    if src_depth != virt_depth:
        new_ups = '../' * virt_depth if virt_depth else ''

        def replace_rel(m: re.Match) -> str:
            ups, rest = m.group(1), m.group(2)
            if ups.count('../') == src_depth:
                return f']({new_ups}{rest})'
            return m.group(0)  # leave unchanged

        text = REL_LINK_RE.sub(replace_rel, text)

    return text


# ---------------------------------------------------------------------------
# Nav order from mkdocs.yml (used for prev/next generation)
# ---------------------------------------------------------------------------

NAV_PAGES_EN = [
    "index.md",
    "docs/getting-started.md",
    "docs/PTO-Virtual-ISA-Manual.md",
    "manual/index.md",
    "manual/01-overview.md",
    "manual/02-machine-model.md",
    "manual/03-state-and-types.md",
    "manual/04-tiles-and-globaltensor.md",
    "manual/05-synchronization.md",
    "manual/06-assembly.md",
    "manual/07-instructions.md",
    "manual/08-programming.md",
    "manual/09-virtual-isa-and-ir.md",
    "manual/10-bytecode-and-toolchain.md",
    "manual/11-memory-ordering-and-consistency.md",
    "manual/12-backend-profiles-and-conformance.md",
    "manual/appendix-a-glossary.md",
    "manual/appendix-b-instruction-contract-template.md",
    "manual/appendix-c-diagnostics-taxonomy.md",
    "manual/appendix-d-instruction-family-matrix.md",
    "docs/coding/README.md",
    "docs/coding/ProgrammingModel.md",
    "docs/coding/Tile.md",
    "docs/coding/GlobalTensor.md",
    "docs/coding/Scalar.md",
    "docs/coding/Event.md",
    "docs/coding/tutorial.md",
    "docs/coding/tutorials/README.md",
    "docs/coding/tutorials/vec-add.md",
    "docs/coding/tutorials/row-softmax.md",
    "docs/coding/tutorials/gemm.md",
    "docs/coding/opt.md",
    "docs/coding/debug.md",
    "docs/machine/abstract-machine.md",
    "docs/machine/README.md",
    "docs/isa/README.md",
    "docs/isa/conventions.md",
    "docs/assembly/README.md",
    "docs/assembly/PTO-AS.md",
    "docs/assembly/conventions.md",
    "docs/assembly/nonisa-ops.md",
    "docs/assembly/elementwise-ops.md",
    "docs/assembly/tile-scalar-ops.md",
    "docs/assembly/axis-ops.md",
    "docs/assembly/memory-ops.md",
    "docs/assembly/matrix-ops.md",
    "docs/assembly/data-movement-ops.md",
    "docs/assembly/complex-ops.md",
    "docs/assembly/manual-binding-ops.md",
    "docs/assembly/scalar-arith-ops.md",
    "docs/assembly/control-flow-ops.md",
    "docs/PTOISA.md",
    "docs/isa/TSYNC.md",
    "docs/isa/TASSIGN.md",
    "docs/isa/TSETHF32MODE.md",
    "docs/isa/TSETTF32MODE.md",
    "docs/isa/TSETFMATRIX.md",
    "docs/isa/TSET_IMG2COL_RPT.md",
    "docs/isa/TSET_IMG2COL_PADDING.md",
    "docs/isa/TADD.md", "docs/isa/TABS.md", "docs/isa/TAND.md",
    "docs/isa/TOR.md", "docs/isa/TSUB.md", "docs/isa/TMUL.md",
    "docs/isa/TMIN.md", "docs/isa/TMAX.md", "docs/isa/TCMP.md",
    "docs/isa/TDIV.md", "docs/isa/TSHL.md", "docs/isa/TSHR.md",
    "docs/isa/TXOR.md", "docs/isa/TLOG.md", "docs/isa/TRECIP.md",
    "docs/isa/TPRELU.md", "docs/isa/TADDC.md", "docs/isa/TSUBC.md",
    "docs/isa/TCVT.md", "docs/isa/TSEL.md", "docs/isa/TRSQRT.md",
    "docs/isa/TSQRT.md", "docs/isa/TEXP.md", "docs/isa/TNOT.md",
    "docs/isa/TRELU.md", "docs/isa/TNEG.md", "docs/isa/TREM.md",
    "docs/isa/TFMOD.md",
    "docs/isa/TEXPANDS.md", "docs/isa/TCMPS.md", "docs/isa/TSELS.md",
    "docs/isa/TMINS.md", "docs/isa/TADDS.md", "docs/isa/TAXPY.md", "docs/isa/TSUBS.md",
    "docs/isa/TDIVS.md", "docs/isa/TMULS.md", "docs/isa/TFMODS.md",
    "docs/isa/TREMS.md", "docs/isa/TMAXS.md", "docs/isa/TANDS.md",
    "docs/isa/TORS.md", "docs/isa/TSHLS.md", "docs/isa/TSHRS.md",
    "docs/isa/TXORS.md", "docs/isa/TLRELU.md", "docs/isa/TADDSC.md",
    "docs/isa/TSUBSC.md",
    "docs/isa/TROWSUM.md", "docs/isa/TROWPROD.md", "docs/isa/TCOLSUM.md",
    "docs/isa/TCOLPROD.md", "docs/isa/TCOLMAX.md", "docs/isa/TROWMAX.md",
    "docs/isa/TROWMIN.md", "docs/isa/TCOLMIN.md", "docs/isa/TROWEXPAND.md",
    "docs/isa/TROWEXPANDDIV.md", "docs/isa/TROWEXPANDMUL.md",
    "docs/isa/TROWEXPANDSUB.md", "docs/isa/TROWEXPANDADD.md",
    "docs/isa/TROWEXPANDMAX.md", "docs/isa/TROWEXPANDMIN.md",
    "docs/isa/TROWEXPANDEXPDIF.md", "docs/isa/TCOLEXPAND.md",
    "docs/isa/TCOLEXPANDDIV.md", "docs/isa/TCOLEXPANDMUL.md",
    "docs/isa/TCOLEXPANDADD.md", "docs/isa/TCOLEXPANDMAX.md",
    "docs/isa/TCOLEXPANDMIN.md", "docs/isa/TCOLEXPANDSUB.md",
    "docs/isa/TCOLEXPANDEXPDIF.md",
    "docs/isa/TLOAD.md", "docs/isa/TPREFETCH.md", "docs/isa/TSTORE.md",
    "docs/isa/TSTORE_FP.md", "docs/isa/MGATHER.md", "docs/isa/MSCATTER.md",
    "docs/isa/TMATMUL.md", "docs/isa/TMATMUL_ACC.md", "docs/isa/TMATMUL_BIAS.md",
    "docs/isa/TMATMUL_MX.md", "docs/isa/TGEMV.md", "docs/isa/TGEMV_ACC.md",
    "docs/isa/TGEMV_BIAS.md", "docs/isa/TGEMV_MX.md",
    "docs/isa/TMOV.md", "docs/isa/TMOV_FP.md", "docs/isa/TEXTRACT.md",
    "docs/isa/TEXTRACT_FP.md", "docs/isa/TINSERT.md", "docs/isa/TINSERT_FP.md",
    "docs/isa/TFILLPAD.md", "docs/isa/TFILLPAD_INPLACE.md",
    "docs/isa/TFILLPAD_EXPAND.md", "docs/isa/TRESHAPE.md", "docs/isa/TALIAS.md",
    "docs/isa/TSUBVIEW.md", "docs/isa/TCONCAT.md", "docs/isa/TTRANS.md",
    "docs/isa/TPACK.md", "docs/isa/TIMG2COL.md",
    "docs/isa/TGATHER.md", "docs/isa/TGATHERB.md", "docs/isa/TSCATTER.md",
    "docs/isa/TCI.md", "docs/isa/TTRI.md", "docs/isa/TPARTADD.md",
    "docs/isa/TPARTMUL.md", "docs/isa/TPARTMAX.md", "docs/isa/TPARTMIN.md",
    "docs/isa/TSORT32.md", "docs/isa/TMRGSORT.md", "docs/isa/TQUANT.md",
    "docs/isa/TDEQUANT.md", "docs/isa/TPUSH.md", "docs/isa/TPOP.md",
    "docs/isa/TFREE.md", "docs/isa/THISTOGRAM.md", "docs/isa/TPRINT.md",
    "docs/isa/comm/README.md",
    "docs/isa/comm/TPUT.md", "docs/isa/comm/TGET.md",
    "docs/isa/comm/TPUT_ASYNC.md", "docs/isa/comm/TGET_ASYNC.md",
    "docs/isa/comm/TNOTIFY.md", "docs/isa/comm/TWAIT.md", "docs/isa/comm/TTEST.md",
    "docs/isa/comm/TGATHER.md", "docs/isa/comm/TSCATTER.md",
    "docs/isa/comm/TREDUCE.md", "docs/isa/comm/TBROADCAST.md",
    "docs/reference/pto-intrinsics-header.md",
    "manual/isa-reference.md",
    # Examples & Kernels
    "kernels/README.md",
    "kernels/manual/a2a3/gemm_performance/README.md",
    "kernels/manual/common/flash_atten/README.md",
    "demos/baseline/add/README.md",
    "demos/baseline/gemm_basic/README.md",
    "tests/README.md",
    "tests/script/README.md",
    # Documentation
    "docs/README.md",
    "docs/website.md",
    # Full index
    "all-pages.md",
]


def _md_to_url(md_path: str) -> str:
    """Convert a virtual .md path to the MkDocs site URL path.

    MkDocs converts:
      - ``foo/index.md``  -> ``/foo/``
      - ``foo/README.md`` -> ``/foo/``   (README treated as directory index)
      - ``index.md``      -> ``/``
      - ``README.md``     -> ``/``
      - ``foo/bar.md``    -> ``/foo/bar/``
    """
    p = Path(md_path)
    if p.name in ("index.md", "README.md"):
        parent = p.parent.as_posix().lstrip("./")
        url = "/" + parent + "/" if parent else "/"
    else:
        url = "/" + p.with_suffix("").as_posix().lstrip("./") + "/"
    # normalise double-slash at root
    if url == "//":
        url = "/"
    return url


def _en_url_to_zh_url(en_url: str) -> str | None:
    """Best-effort mapping: English URL -> Chinese URL.

    Returns None if we cannot determine the zh counterpart.
    """
    # root index: / -> /index_zh/
    if en_url == "/":
        return "/index_zh/"
    # strip trailing slash for manipulation
    base = en_url.rstrip("/")
    # manual index: /manual -> /manual/index_zh
    if base == "/manual":
        return "/manual/index_zh/"
    # README pages: last segment is a known directory name
    last = base.rsplit("/", 1)[-1]
    if last in README_DIRS:
        return en_url.rstrip("/") + "/README_zh/"
    # general page: append _zh
    return base + "_zh/"


def _generate_lang_map(nav_pages: list[str]) -> dict:
    """Build a mapping dict for use by the language switcher JS.

    Structure::

        {
          "en_to_zh": { "/manual/01-overview/": "/manual/01-overview_zh/", ... },
          "zh_to_en": { "/manual/01-overview_zh/": "/manual/01-overview/", ... },
          "nav": [
            { "en": "/manual/01-overview/", "zh": "/manual/01-overview_zh/",
              "prev_en": "/manual/", "prev_zh": "/manual/index_zh/",
              "next_en": "/manual/02-machine-model/",
              "next_zh": "/manual/02-machine-model_zh/" },
            ...
          ]
        }
    """
    en_urls = [_md_to_url(p) for p in nav_pages]
    en_to_zh: dict[str, str] = {}
    zh_to_en: dict[str, str] = {}

    for en in en_urls:
        zh = _en_url_to_zh_url(en)
        if zh:
            en_to_zh[en] = zh
            zh_to_en[zh] = en

    nav_entries = []
    for i, en in enumerate(en_urls):
        zh = en_to_zh.get(en)
        prev_en = en_urls[i - 1] if i > 0 else None
        next_en = en_urls[i + 1] if i < len(en_urls) - 1 else None
        entry = {
            "en": en,
            "zh": zh,
            "prev_en": prev_en,
            "prev_zh": en_to_zh.get(prev_en) if prev_en else None,
            "next_en": next_en,
            "next_zh": en_to_zh.get(next_en) if next_en else None,
        }
        nav_entries.append(entry)

    return {"en_to_zh": en_to_zh, "zh_to_en": zh_to_en, "nav": nav_entries}


# ---------------------------------------------------------------------------
# Helpers used by main()
# ---------------------------------------------------------------------------

def _extract_first_heading(md_path: Path) -> str:
    """Return the text of the first Markdown heading in *md_path*, or the stem."""
    try:
        text = md_path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return md_path.stem
    for line in text.splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return md_path.stem


@dataclass
class IsaReferenceIndexConfig:
    """Configuration for generating an ISA reference index page."""
    out_path: str
    isa_pages: list[tuple[str, str]]
    heading: str
    preamble: str
    section_heading: str
    empty_msg: str


def _write_isa_reference_index(config: IsaReferenceIndexConfig) -> None:
    """Write a generated ISA reference index page to *config.out_path*."""
    with mkdocs_gen_files.open(config.out_path, "w") as f:
        f.write(f"{config.heading}\n\n")
        f.write(config.preamble)
        if not config.isa_pages:
            f.write(config.empty_msg)
        else:
            f.write(f"{config.section_heading}\n\n")
            for instr, _ in config.isa_pages:
                link = f"../docs/isa/{instr}.md"
                bare = instr[:-3] if instr.endswith("_zh") else instr
                display = bare.split("/")[-1]
                f.write(f"- [{display}]({link})\n")
            f.write("\n")


def _format_section_entry(rel: str, top: str) -> str:
    """Return a markdown list entry for a single page in a section."""
    label = rel if top == "(root)" else rel[len(top) + 1:]
    return f"- [{label}]({rel})\n"


def _write_sections(f, sections: dict[str, list[str]]) -> None:
    """Write all section headings and page entries to an open file handle."""
    for top in sorted(sections.keys()):
        f.write(f"## {top}\n\n")
        for rel in sections[top]:
            f.write(_format_section_entry(rel, top))
        f.write("\n")


def _write_all_pages_index(
    out_path: str,
    sections: dict[str, list[str]],
    heading: str,
    preamble: str,
    empty_msg: str,
) -> None:
    """Write a generated all-pages index to *out_path*."""
    with mkdocs_gen_files.open(out_path, "w") as f:
        f.write(f"{heading}\n\n")
        f.write(preamble)
        if not sections:
            f.write(empty_msg)
        else:
            _write_sections(f, sections)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    copied_md: list[str] = []

    mkdocs_src = REPO_ROOT / "docs" / "mkdocs" / "src"

    # Step 1: Process hand-written files under docs/mkdocs/src/.
    # These files use root-absolute links (e.g. /docs/isa/TADD.md) so they
    # work when browsing the repo statically (GitHub/Gitee). At build time
    # we rewrite them to relative paths for MkDocs, and place them at their
    # virtual path (stripping the docs/mkdocs/src/ prefix).
    #
    # IMPORTANT: We must NOT recurse into docs/mkdocs/src/docs/mkdocs/src/
    # (stale nested copies from previous builds). We skip any path that,
    # relative to mkdocs_src, starts with "docs/mkdocs/" to avoid that.
    for src in mkdocs_src.rglob("*.md"):
        virtual_path = src.relative_to(mkdocs_src).as_posix()  # e.g. manual/01-overview.md
        # Skip stale nested docs/mkdocs/src/ directories that may exist on disk
        # from previous builds (mkdocs_gen_files writes to a temp dir, but the
        # docs_dir itself may have leftover files if docs_dir == src/).
        if virtual_path.startswith("docs/mkdocs/"):
            continue
        text = src.read_text(encoding="utf-8-sig", errors="replace")
        # Strip any stale "Generated from" header left by a previous build.
        text = _strip_generated_header(text)
        text = _rewrite_rel_imgs_for_build(text, virtual_path)
        text = _rewrite_links_for_build(text, virtual_path)
        with mkdocs_gen_files.open(virtual_path, "w") as f:
            # Do not add a generated header for hand-written files under
            # docs/mkdocs/src/manual/ or the root index pages — those are
            # source files, not build artefacts, and the comment would
            # pollute the originals.
            if not virtual_path.startswith("manual/") and virtual_path not in ("index.md", "index_zh.md"):
                f.write(f"<!-- Generated from `docs/mkdocs/src/{virtual_path}` -->\n\n")
            f.write(text)
        copied_md.append(virtual_path)

    # Step 2: Mirror all other repo markdown files preserving their paths.
    for src in REPO_ROOT.rglob("*.md"):
        rel = src.relative_to(REPO_ROOT).as_posix()
        if _should_skip(rel):
            continue
        # Use utf-8-sig to automatically remove BOM if present
        text = src.read_text(encoding="utf-8-sig", errors="replace")
        # Rewrite relative <img src="..."> paths for all mirrored files.
        text = _rewrite_rel_imgs_for_build(text, rel)
        # Rewrite mkdocs/src/... links to root-absolute /... links, then
        # let _rewrite_links_for_build convert them to correct relative paths.
        if MKDOCS_SRC_LINK_RE.search(text):
            text = MKDOCS_SRC_LINK_RE.sub(r'](/\1)', text)
            text = _rewrite_links_for_build(text, rel)
        with mkdocs_gen_files.open(rel, "w") as f:
            f.write(f"<!-- Generated from `{rel}` -->\n\n")
            f.write(text)
        copied_md.append(rel)

    # Generate per-instruction reference indexes for docs/isa/*.md.
    isa_dir = REPO_ROOT / "docs" / "isa"
    isa_pages_en: list[tuple[str, str]] = []
    isa_pages_zh: list[tuple[str, str]] = []

    if isa_dir.exists():
        for p in sorted(isa_dir.glob("*.md")):
            if p.name in ("README.md", "README_zh.md", "conventions.md", "conventions_zh.md"):
                continue
            stem = p.stem
            title = _extract_first_heading(p)
            if stem.endswith("_zh"):
                isa_pages_zh.append((stem, title))
            else:
                isa_pages_en.append((stem, title))
        # Also include comm sub-directory instructions
        comm_dir = isa_dir / "comm"
        if comm_dir.exists():
            for p in sorted(comm_dir.glob("*.md")):
                if p.name in ("README.md", "README_zh.md"):
                    continue
                stem = p.stem
                title = _extract_first_heading(p)
                if stem.endswith("_zh"):
                    isa_pages_zh.append(("comm/" + stem, title))
                else:
                    isa_pages_en.append(("comm/" + stem, title))

    _write_isa_reference_index(IsaReferenceIndexConfig(
        out_path="manual/isa-reference.md",
        isa_pages=isa_pages_en,
        heading="# Instruction Reference Pages",
        preamble=(
            "This page is generated at build time.\n\n"
            "- Instruction index: `docs/isa/README.md`\n"
            "- ISA conventions: `docs/isa/conventions.md`\n\n"
        ),
        section_heading="## All instructions",
        empty_msg="No English instruction pages were found under `docs/isa/`.\n",
    ))

    _write_isa_reference_index(IsaReferenceIndexConfig(
        out_path="manual/isa-reference_zh.md",
        isa_pages=isa_pages_zh,
        heading="# 指令参考页面（全量）",
        preamble=(
            "本页在构建站点时自动生成。\n\n"
            "- 指令索引：`docs/isa/README_zh.md`\n"
            "- ISA 通用约定：`docs/isa/conventions_zh.md`\n\n"
        ),
        section_heading="## 全部指令",
        empty_msg="未在 `docs/isa/` 下发现中文指令页面。\n",
    ))

    # Generate a simple index page that links to all mirrored markdown.
    all_md = sorted(set(copied_md))
    sections: dict[str, list[str]] = {}
    sections_zh: dict[str, list[str]] = {}

    for rel in all_md:
        top = rel.split("/", 1)[0] if "/" in rel else "(root)"
        sections.setdefault(top, []).append(rel)
        if "_zh.md" in rel or rel.endswith("_zh/index.md"):
            sections_zh.setdefault(top, []).append(rel)

    _write_all_pages_index(
        out_path="all-pages.md",
        sections=sections,
        heading="# All Markdown Pages",
        preamble="This page is generated at build time and lists markdown files mirrored into the site.\n\n",
        empty_msg="",
    )

    _write_all_pages_index(
        out_path="all-pages_zh.md",
        sections=sections_zh,
        heading="# 所有 Markdown 页面",
        preamble="本页面在构建时自动生成，列出了站点中镜像的所有中文 markdown 文件。\n\n",
        empty_msg="未找到中文页面。\n",
    )

    # Generate lang-map.json for zero-latency language switching.
    lang_map = _generate_lang_map(NAV_PAGES_EN)
    with mkdocs_gen_files.open("lang-map.json", "w") as f:
        json.dump(lang_map, f, ensure_ascii=False, separators=(",", ":"))

    # Mirror commonly referenced doc assets (images) so docs render cleanly.
    for src in REPO_ROOT.rglob("*"):
        if not src.is_file():
            continue
        if src.suffix.lower() not in ASSET_EXTS:
            continue
        rel = src.relative_to(REPO_ROOT).as_posix()
        if _should_skip(rel):
            continue
        with mkdocs_gen_files.open(rel, "wb") as f:
            f.write(src.read_bytes())


main()
 
