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
MkDocs build-time generator for the PTO ISA manual.

MkDocs configuration and site chrome live under `docs/mkdocs/`. Authored PTO ISA
content lives in the canonical documentation tree, primarily `docs/isa/`.
This script publishes only the curated canonical sources that belong in the
manual site; it does not mirror the repository into `docs/mkdocs/src/`.

Source-of-truth rule:
- `docs/isa/` is the canonical authored ISA manual and instruction reference.
- `docs/mkdocs/src/` is only for MkDocs-owned landing pages and static assets.
- generated pages are build outputs and must not be committed as source files.
"""

from __future__ import annotations

import json
import posixpath
import re
import shutil
from pathlib import Path

import mkdocs_gen_files


REPO_ROOT = Path(__file__).resolve().parents[2]

SKIP_PREFIXES = (
    ".git/",
    ".github/",
    ".gitcode/",
    ".claude/",
    ".codex/",
    ".venv/",
    ".venv-mkdocs/",
    "agents/",
    "site/",
    "site_zh/",
    "build/",
    "build_tests/",
    ".idea/",
    ".vscode/",
)

SKIP_CONTAINS = ("/__pycache__/", "/CMakeFiles/")

SKIP_EXACT = {
    "AGENTS.md",
    "CONTRIBUTING.md",
    "CONTRIBUTING_zh.md",
    "ReleaseNote.md",
    "ReleaseNote_zh.md",
    "SECURITY.md",
    "SECURITY_zh.md",
    "docs/menu_ops_development.md",
}

ASSET_EXTS = {".svg", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bnf"}

PUBLISHED_MD_PREFIXES = ("docs/isa/",)

PUBLISHED_MD_EXACT = {
    "docs/getting-started.md",
    "docs/getting-started_zh.md",
}

PUBLISHED_ASSET_PREFIXES = ("docs/figures/",)

HANDWRITTEN_MKDOCS_MD = {"index.md", "index_zh.md"}

GENERATED_MKDOCS_ROOTS = (
    ".claude",
    ".codex",
    "agents",
    "assembly",
    "cmake",
    "demos",
    "docs",
    "include",
    "kernels",
    "manual",
    "scripts",
    "tests",
)

GENERATED_MKDOCS_FILES = (
    "AGENTS.md",
    "CONTRIBUTING.md",
    "CONTRIBUTING_zh.md",
    "README.md",
    "README_zh.md",
    "ReleaseNote.md",
    "ReleaseNote_zh.md",
    "SECURITY.md",
    "SECURITY_zh.md",
    "all-pages.md",
    "all-pages_zh.md",
    "lang-map.json",
)

# Directory names whose README is the canonical index page.
# Used by _en_url_to_zh_url to map e.g. /docs/isa/ -> /docs/isa/README_zh/.
README_DIRS = {
    "coding",
    "isa",
    "machine",
    "assembly",
    "docs",
    "kernels",
    "tests",
    "demos",
    "scripts",
    "include",
    "cmake",
    "reference",
    "tutorials",
    "script",
    "package",
    "custom",
    "baseline",
    "add",
    "gemm_basic",
    "flash_atten",
    "gemm_performance",
    "a2a3",
    "a5",
    "kirin9030",
    "npu",
    "pto",
    "comm",
    "introduction",
    "programming-model",
    "machine-model",
    "memory-model",
    "state-and-types",
    "syntax-and-operands",
    "instruction-families",
}


def _should_skip(rel_posix: str) -> bool:
    if rel_posix.startswith("docs/mkdocs/"):
        return True
    if rel_posix.endswith("/mkdocs.yml"):
        return True
    if rel_posix in SKIP_EXACT:
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


def _is_published_md(rel_posix: str) -> bool:
    return rel_posix in PUBLISHED_MD_EXACT or rel_posix.startswith(PUBLISHED_MD_PREFIXES)


def _is_published_asset(rel_posix: str) -> bool:
    return rel_posix.startswith(PUBLISHED_ASSET_PREFIXES)


ABS_LINK_RE = re.compile(r"\]\(/((?!http)[^)]+)\)")
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
    page_dir = Path(src_rel).with_suffix("").as_posix()  # e.g. "docs/getting-started"

    def replace(m: re.Match) -> str:
        prefix, img_path, suffix = m.group(1), m.group(2), m.group(3)
        # Resolve image to repo-relative path.
        repo_img = (src_dir + "/" + img_path) if (src_dir and src_dir != ".") else img_path
        # Normalize (handle any ../ in original img_path).
        repo_img = posixpath.normpath(repo_img)
        # Compute relative path from page_dir to repo_img.
        rel = posixpath.relpath(repo_img, page_dir)  # e.g. ../figures/pto_logo.svg
        return f"{prefix}{rel}{suffix}"

    return REL_IMG_RE.sub(replace, text)


# Matches a relative markdown link that starts with one or more "../" components.
# Group 1: the "../" prefix (one or more), Group 2: the rest of the target.
REL_LINK_RE = re.compile(r"\]\((\.\./+)([^)]+)\)")

# Matches repo-relative links into docs/mkdocs/src/.
# These appear in repo-level docs so they resolve correctly during static browsing,
# but must be rewritten to root-absolute paths at MkDocs build time.
MKDOCS_SRC_LINK_RE = re.compile(r"\]\(mkdocs/src/([^)]+)\)")

# Strip stale "<!-- Generated from ... -->" header lines that accumulate on
# repeated builds when docs_dir is the same directory as the source files.
_GENERATED_HEADER_RE = re.compile(r"^(?:<!-- Generated from `[^`]*` -->\s*\n\n?)+", re.MULTILINE)


def _strip_generated_header(text: str) -> str:
    """Remove any leading '<!-- Generated from ... -->' comment blocks."""
    return _GENERATED_HEADER_RE.sub("", text)


def _rewrite_links_for_build(text: str, virtual_path: str) -> str:
    """Rewrite links in a hand-written docs/mkdocs/src/ file so they resolve
    correctly from the MkDocs virtual page URL.

    Two kinds of links are rewritten:

    1. Root-absolute links like /docs/isa/tile/ops/elementwise-tile-tile/tadd.md  ->  ../docs/isa/tile/ops/elementwise-tile-tile/tadd.md
       These are written with a leading '/' so they work when browsing the
       repo on GitHub/Gitee; at build time they need to be relative.

    2. Relative links whose "../" depth is wrong for the virtual path.
       Example: hand-written file lives at docs/mkdocs/src/index.md but is
       published as index.md. The function adjusts any repo-relative links
       that were written from the physical source path.

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
                      "index.md".
    """
    # Depth of virtual page's parent directory.
    virt_depth = len(Path(virtual_path).parent.parts)  # e.g. 1 for "manual/foo.md"

    # Depth of the source file inside docs/mkdocs/src/.
    src_depth = len((Path("docs") / "mkdocs" / "src" / virtual_path).parent.parts)

    # --- Pass 1: root-absolute links /foo/bar  ->  (../)*virt_depth foo/bar ---
    prefix_abs = "../" * virt_depth if virt_depth else ""

    def replace_abs(m: re.Match) -> str:
        return f"]({prefix_abs}{m.group(1)})"

    text = ABS_LINK_RE.sub(replace_abs, text)

    # --- Pass 2: relative links with wrong ../ depth ---
    # The author wrote the link relative to the *repo* source file location
    # (src_depth levels deep).  We need it relative to the virtual page
    # (virt_depth levels deep).  We only touch links whose leading "../"
    # count equals src_depth (exactly what the author would write to reach
    # the repo root from the source file).
    if src_depth != virt_depth:
        new_ups = "../" * virt_depth if virt_depth else ""

        def replace_rel(m: re.Match) -> str:
            ups, rest = m.group(1), m.group(2)
            if ups.count("../") == src_depth:
                return f"]({new_ups}{rest})"
            return m.group(0)  # leave unchanged

        text = REL_LINK_RE.sub(replace_rel, text)

    return text


# ---------------------------------------------------------------------------
# Nav order from mkdocs.yml (used for prev/next generation)
# ---------------------------------------------------------------------------

TILE_REFERENCE_PAGES = [
    "docs/isa/tile/README.md",
    "docs/isa/tile/sync-and-config.md",
    "docs/isa/tile/ops/sync-and-config/tsync.md",
    "docs/isa/tile/ops/sync-and-config/tassign.md",
    "docs/isa/tile/ops/sync-and-config/talias.md",
    "docs/isa/tile/ops/sync-and-config/setfmatrix.md",
    "docs/isa/tile/ops/sync-and-config/settf32mode.md",
    "docs/isa/tile/ops/sync-and-config/set-img2col-rpt.md",
    "docs/isa/tile/ops/sync-and-config/set-img2col-padding.md",
    "docs/isa/tile/ops/sync-and-config/subview.md",
    "docs/isa/tile/ops/sync-and-config/get-scale-addr.md",
    "docs/isa/tile/elementwise-tile-tile.md",
    "docs/isa/tile/ops/elementwise-tile-tile/tadd.md",
    "docs/isa/tile/ops/elementwise-tile-tile/tabs.md",
    "docs/isa/tile/ops/elementwise-tile-tile/tand.md",
    "docs/isa/tile/ops/elementwise-tile-tile/tor.md",
    "docs/isa/tile/ops/elementwise-tile-tile/tsub.md",
    "docs/isa/tile/ops/elementwise-tile-tile/tmul.md",
    "docs/isa/tile/ops/elementwise-tile-tile/tmin.md",
    "docs/isa/tile/ops/elementwise-tile-tile/tmax.md",
    "docs/isa/tile/ops/elementwise-tile-tile/tcmp.md",
    "docs/isa/tile/ops/elementwise-tile-tile/tdiv.md",
    "docs/isa/tile/ops/elementwise-tile-tile/tshl.md",
    "docs/isa/tile/ops/elementwise-tile-tile/tshr.md",
    "docs/isa/tile/ops/elementwise-tile-tile/txor.md",
    "docs/isa/tile/ops/elementwise-tile-tile/tlog.md",
    "docs/isa/tile/ops/elementwise-tile-tile/trecip.md",
    "docs/isa/tile/ops/elementwise-tile-tile/tprelu.md",
    "docs/isa/tile/ops/elementwise-tile-tile/taddc.md",
    "docs/isa/tile/ops/elementwise-tile-tile/tsubc.md",
    "docs/isa/tile/ops/elementwise-tile-tile/tcvt.md",
    "docs/isa/tile/ops/elementwise-tile-tile/tsel.md",
    "docs/isa/tile/ops/elementwise-tile-tile/trsqrt.md",
    "docs/isa/tile/ops/elementwise-tile-tile/tsqrt.md",
    "docs/isa/tile/ops/elementwise-tile-tile/texp.md",
    "docs/isa/tile/ops/elementwise-tile-tile/tnot.md",
    "docs/isa/tile/ops/elementwise-tile-tile/trelu.md",
    "docs/isa/tile/ops/elementwise-tile-tile/tneg.md",
    "docs/isa/tile/ops/elementwise-tile-tile/trem.md",
    "docs/isa/tile/ops/elementwise-tile-tile/tfmod.md",
    "docs/isa/tile/elementwise-tile-tile.md",
    "docs/isa/tile/tile-scalar-and-immediate.md",
    "docs/isa/tile/ops/tile-scalar-and-immediate/texpands.md",
    "docs/isa/tile/ops/tile-scalar-and-immediate/taxpy.md",
    "docs/isa/tile/ops/tile-scalar-and-immediate/tcmps.md",
    "docs/isa/tile/ops/tile-scalar-and-immediate/tsels.md",
    "docs/isa/tile/ops/tile-scalar-and-immediate/tmins.md",
    "docs/isa/tile/ops/tile-scalar-and-immediate/tadds.md",
    "docs/isa/tile/ops/tile-scalar-and-immediate/tsubs.md",
    "docs/isa/tile/ops/tile-scalar-and-immediate/tdivs.md",
    "docs/isa/tile/ops/tile-scalar-and-immediate/tmuls.md",
    "docs/isa/tile/ops/tile-scalar-and-immediate/tfmods.md",
    "docs/isa/tile/ops/tile-scalar-and-immediate/trems.md",
    "docs/isa/tile/ops/tile-scalar-and-immediate/tmaxs.md",
    "docs/isa/tile/ops/tile-scalar-and-immediate/tands.md",
    "docs/isa/tile/ops/tile-scalar-and-immediate/tors.md",
    "docs/isa/tile/ops/tile-scalar-and-immediate/tshls.md",
    "docs/isa/tile/ops/tile-scalar-and-immediate/tshrs.md",
    "docs/isa/tile/ops/tile-scalar-and-immediate/txors.md",
    "docs/isa/tile/ops/tile-scalar-and-immediate/tlrelu.md",
    "docs/isa/tile/ops/tile-scalar-and-immediate/taddsc.md",
    "docs/isa/tile/ops/tile-scalar-and-immediate/tsubsc.md",
    "docs/isa/tile/reduce-and-expand.md",
    "docs/isa/tile/ops/reduce-and-expand/trowsum.md",
    "docs/isa/tile/ops/reduce-and-expand/tcolsum.md",
    "docs/isa/tile/ops/reduce-and-expand/trowprod.md",
    "docs/isa/tile/ops/reduce-and-expand/tcolprod.md",
    "docs/isa/tile/ops/reduce-and-expand/tcolmax.md",
    "docs/isa/tile/ops/reduce-and-expand/tcolmin.md",
    "docs/isa/tile/ops/reduce-and-expand/tcolargmax.md",
    "docs/isa/tile/ops/reduce-and-expand/tcolargmin.md",
    "docs/isa/tile/ops/reduce-and-expand/trowmax.md",
    "docs/isa/tile/ops/reduce-and-expand/trowmin.md",
    "docs/isa/tile/ops/reduce-and-expand/trowargmax.md",
    "docs/isa/tile/ops/reduce-and-expand/trowargmin.md",
    "docs/isa/tile/ops/reduce-and-expand/trowexpand.md",
    "docs/isa/tile/ops/reduce-and-expand/trowexpanddiv.md",
    "docs/isa/tile/ops/reduce-and-expand/trowexpandmul.md",
    "docs/isa/tile/ops/reduce-and-expand/trowexpandsub.md",
    "docs/isa/tile/ops/reduce-and-expand/trowexpandadd.md",
    "docs/isa/tile/ops/reduce-and-expand/trowexpandmax.md",
    "docs/isa/tile/ops/reduce-and-expand/trowexpandmin.md",
    "docs/isa/tile/ops/reduce-and-expand/trowexpandexpdif.md",
    "docs/isa/tile/ops/reduce-and-expand/tcolexpand.md",
    "docs/isa/tile/ops/reduce-and-expand/tcolexpanddiv.md",
    "docs/isa/tile/ops/reduce-and-expand/tcolexpandmul.md",
    "docs/isa/tile/ops/reduce-and-expand/tcolexpandadd.md",
    "docs/isa/tile/ops/reduce-and-expand/tcolexpandmax.md",
    "docs/isa/tile/ops/reduce-and-expand/tcolexpandmin.md",
    "docs/isa/tile/ops/reduce-and-expand/tcolexpandsub.md",
    "docs/isa/tile/ops/reduce-and-expand/tcolexpandexpdif.md",
    "docs/isa/tile/memory-and-data-movement.md",
    "docs/isa/tile/ops/memory-and-data-movement/tload.md",
    "docs/isa/tile/ops/memory-and-data-movement/tprefetch.md",
    "docs/isa/tile/ops/memory-and-data-movement/tstore.md",
    "docs/isa/tile/ops/memory-and-data-movement/mgather.md",
    "docs/isa/tile/ops/memory-and-data-movement/mscatter.md",
    "docs/isa/tile/matrix-and-matrix-vector.md",
    "docs/isa/tile/ops/matrix-and-matrix-vector/tgemv-mx.md",
    "docs/isa/tile/ops/matrix-and-matrix-vector/tmatmul-mx.md",
    "docs/isa/tile/ops/matrix-and-matrix-vector/tmatmul.md",
    "docs/isa/tile/ops/matrix-and-matrix-vector/tmatmul-acc.md",
    "docs/isa/tile/ops/matrix-and-matrix-vector/tmatmul-bias.md",
    "docs/isa/tile/ops/matrix-and-matrix-vector/tgemv.md",
    "docs/isa/tile/ops/matrix-and-matrix-vector/tgemv-acc.md",
    "docs/isa/tile/ops/matrix-and-matrix-vector/tgemv-bias.md",
    "docs/isa/tile/layout-and-rearrangement.md",
    "docs/isa/tile/ops/layout-and-rearrangement/textract.md",
    "docs/isa/tile/ops/layout-and-rearrangement/timg2col.md",
    "docs/isa/tile/ops/layout-and-rearrangement/tinsert.md",
    "docs/isa/tile/ops/layout-and-rearrangement/tfillpad.md",
    "docs/isa/tile/ops/layout-and-rearrangement/tfillpad-inplace.md",
    "docs/isa/tile/ops/layout-and-rearrangement/tfillpad-expand.md",
    "docs/isa/tile/ops/layout-and-rearrangement/tmov.md",
    "docs/isa/tile/ops/layout-and-rearrangement/tconcat.md",
    "docs/isa/tile/ops/layout-and-rearrangement/tpack.md",
    "docs/isa/tile/ops/layout-and-rearrangement/treshape.md",
    "docs/isa/tile/ops/layout-and-rearrangement/ttrans.md",
    "docs/isa/tile/irregular-and-complex.md",
    "docs/isa/tile/ops/irregular-and-complex/tprint.md",
    "docs/isa/tile/ops/irregular-and-complex/tmrgsort.md",
    "docs/isa/tile/ops/irregular-and-complex/tsort32.md",
    "docs/isa/tile/ops/irregular-and-complex/tgather.md",
    "docs/isa/tile/ops/irregular-and-complex/tgatherb.md",
    "docs/isa/tile/ops/irregular-and-complex/tscatter.md",
    "docs/isa/tile/ops/irregular-and-complex/tci.md",
    "docs/isa/tile/ops/irregular-and-complex/ttri.md",
    "docs/isa/tile/ops/irregular-and-complex/tpartadd.md",
    "docs/isa/tile/ops/irregular-and-complex/tpartmul.md",
    "docs/isa/tile/ops/irregular-and-complex/tpartmax.md",
    "docs/isa/tile/ops/irregular-and-complex/tpartmin.md",
    "docs/isa/tile/ops/irregular-and-complex/tquant.md",
    "docs/isa/tile/ops/irregular-and-complex/tdequant.md",
    "docs/isa/tile/ops/irregular-and-complex/trandom.md",
    "docs/isa/tile/ops/irregular-and-complex/thistogram.md",
]

VECTOR_REFERENCE_PAGES = [
    "docs/isa/vector/README.md",
    "docs/isa/vector/vector-load-store.md",
    "docs/isa/vector/ops/vector-load-store/vlds.md",
    "docs/isa/vector/ops/vector-load-store/vldas.md",
    "docs/isa/vector/ops/vector-load-store/vldus.md",
    "docs/isa/vector/ops/vector-load-store/vldx2.md",
    "docs/isa/vector/ops/vector-load-store/vsld.md",
    "docs/isa/vector/ops/vector-load-store/vsldb.md",
    "docs/isa/vector/ops/vector-load-store/vgather2.md",
    "docs/isa/vector/ops/vector-load-store/vgatherb.md",
    "docs/isa/vector/ops/vector-load-store/vgather2-bc.md",
    "docs/isa/vector/ops/vector-load-store/vsts.md",
    "docs/isa/vector/ops/vector-load-store/vstx2.md",
    "docs/isa/vector/ops/vector-load-store/vsst.md",
    "docs/isa/vector/ops/vector-load-store/vsstb.md",
    "docs/isa/vector/ops/vector-load-store/vscatter.md",
    "docs/isa/vector/ops/vector-load-store/vsta.md",
    "docs/isa/vector/ops/vector-load-store/vstas.md",
    "docs/isa/vector/ops/vector-load-store/vstar.md",
    "docs/isa/vector/ops/vector-load-store/vstu.md",
    "docs/isa/vector/ops/vector-load-store/vstus.md",
    "docs/isa/vector/ops/vector-load-store/vstur.md",
    "docs/isa/vector/predicate-and-materialization.md",
    "docs/isa/vector/ops/predicate-and-materialization/vbr.md",
    "docs/isa/vector/ops/predicate-and-materialization/vdup.md",
    "docs/isa/vector/unary-vector-ops.md",
    "docs/isa/vector/ops/unary-vector-ops/vabs.md",
    "docs/isa/vector/ops/unary-vector-ops/vneg.md",
    "docs/isa/vector/ops/unary-vector-ops/vexp.md",
    "docs/isa/vector/ops/unary-vector-ops/vln.md",
    "docs/isa/vector/ops/unary-vector-ops/vsqrt.md",
    "docs/isa/vector/ops/unary-vector-ops/vrsqrt.md",
    "docs/isa/vector/ops/unary-vector-ops/vrec.md",
    "docs/isa/vector/ops/unary-vector-ops/vrelu.md",
    "docs/isa/vector/ops/unary-vector-ops/vnot.md",
    "docs/isa/vector/ops/unary-vector-ops/vbcnt.md",
    "docs/isa/vector/ops/unary-vector-ops/vcls.md",
    "docs/isa/vector/ops/unary-vector-ops/vmov.md",
    "docs/isa/vector/binary-vector-ops.md",
    "docs/isa/vector/ops/binary-vector-ops/vadd.md",
    "docs/isa/vector/ops/binary-vector-ops/vsub.md",
    "docs/isa/vector/ops/binary-vector-ops/vmul.md",
    "docs/isa/vector/ops/binary-vector-ops/vdiv.md",
    "docs/isa/vector/ops/binary-vector-ops/vmax.md",
    "docs/isa/vector/ops/binary-vector-ops/vmin.md",
    "docs/isa/vector/ops/binary-vector-ops/vand.md",
    "docs/isa/vector/ops/binary-vector-ops/vor.md",
    "docs/isa/vector/ops/binary-vector-ops/vxor.md",
    "docs/isa/vector/ops/binary-vector-ops/vshl.md",
    "docs/isa/vector/ops/binary-vector-ops/vshr.md",
    "docs/isa/vector/ops/binary-vector-ops/vaddc.md",
    "docs/isa/vector/ops/binary-vector-ops/vsubc.md",
    "docs/isa/vector/vec-scalar-ops.md",
    "docs/isa/vector/ops/vec-scalar-ops/vadds.md",
    "docs/isa/vector/ops/vec-scalar-ops/vsubs.md",
    "docs/isa/vector/ops/vec-scalar-ops/vmuls.md",
    "docs/isa/vector/ops/vec-scalar-ops/vmaxs.md",
    "docs/isa/vector/ops/vec-scalar-ops/vmins.md",
    "docs/isa/vector/ops/vec-scalar-ops/vands.md",
    "docs/isa/vector/ops/vec-scalar-ops/vors.md",
    "docs/isa/vector/ops/vec-scalar-ops/vxors.md",
    "docs/isa/vector/ops/vec-scalar-ops/vshls.md",
    "docs/isa/vector/ops/vec-scalar-ops/vshrs.md",
    "docs/isa/vector/ops/vec-scalar-ops/vlrelu.md",
    "docs/isa/vector/ops/vec-scalar-ops/vaddcs.md",
    "docs/isa/vector/ops/vec-scalar-ops/vsubcs.md",
    "docs/isa/vector/conversion-ops.md",
    "docs/isa/vector/ops/conversion-ops/vci.md",
    "docs/isa/vector/ops/conversion-ops/vcvt.md",
    "docs/isa/vector/ops/conversion-ops/vtrc.md",
    "docs/isa/vector/reduction-ops.md",
    "docs/isa/vector/ops/reduction-ops/vcadd.md",
    "docs/isa/vector/ops/reduction-ops/vcmax.md",
    "docs/isa/vector/ops/reduction-ops/vcmin.md",
    "docs/isa/vector/ops/reduction-ops/vcgadd.md",
    "docs/isa/vector/ops/reduction-ops/vcgmax.md",
    "docs/isa/vector/ops/reduction-ops/vcgmin.md",
    "docs/isa/vector/ops/reduction-ops/vcpadd.md",
    "docs/isa/vector/compare-select.md",
    "docs/isa/vector/ops/compare-select/vcmp.md",
    "docs/isa/vector/ops/compare-select/vcmps.md",
    "docs/isa/vector/ops/compare-select/vsel.md",
    "docs/isa/vector/ops/compare-select/vselr.md",
    "docs/isa/vector/ops/compare-select/vselrv2.md",
    "docs/isa/vector/data-rearrangement.md",
    "docs/isa/vector/ops/data-rearrangement/vintlv.md",
    "docs/isa/vector/ops/data-rearrangement/vdintlv.md",
    "docs/isa/vector/ops/data-rearrangement/vslide.md",
    "docs/isa/vector/ops/data-rearrangement/vshift.md",
    "docs/isa/vector/ops/data-rearrangement/vsqz.md",
    "docs/isa/vector/ops/data-rearrangement/vusqz.md",
    "docs/isa/vector/ops/data-rearrangement/vperm.md",
    "docs/isa/vector/ops/data-rearrangement/vpack.md",
    "docs/isa/vector/ops/data-rearrangement/vsunpack.md",
    "docs/isa/vector/ops/data-rearrangement/vzunpack.md",
    "docs/isa/vector/ops/data-rearrangement/vintlvv2.md",
    "docs/isa/vector/ops/data-rearrangement/vdintlvv2.md",
    "docs/isa/vector/sfu-and-dsa-ops.md",
    "docs/isa/vector/ops/sfu-and-dsa-ops/vprelu.md",
    "docs/isa/vector/ops/sfu-and-dsa-ops/vexpdiff.md",
    "docs/isa/vector/ops/sfu-and-dsa-ops/vaddrelu.md",
    "docs/isa/vector/ops/sfu-and-dsa-ops/vsubrelu.md",
    "docs/isa/vector/ops/sfu-and-dsa-ops/vaxpy.md",
    "docs/isa/vector/ops/sfu-and-dsa-ops/vaddreluconv.md",
    "docs/isa/vector/ops/sfu-and-dsa-ops/vmulconv.md",
    "docs/isa/vector/ops/sfu-and-dsa-ops/vmull.md",
    "docs/isa/vector/ops/sfu-and-dsa-ops/vmula.md",
    "docs/isa/vector/ops/sfu-and-dsa-ops/vtranspose.md",
    "docs/isa/vector/ops/sfu-and-dsa-ops/vsort32.md",
    "docs/isa/vector/ops/sfu-and-dsa-ops/vmrgsort.md",
    "docs/isa/vector/shared-arith.md",
    "docs/isa/vector/shared-scf.md",
]

SCALAR_REFERENCE_PAGES = [
    "docs/isa/scalar/README.md",
    "docs/isa/scalar/control-and-configuration.md",
    "docs/isa/scalar/pipeline-sync.md",
    "docs/isa/scalar/ops/pipeline-sync/set-flag.md",
    "docs/isa/scalar/ops/pipeline-sync/wait-flag.md",
    "docs/isa/scalar/ops/pipeline-sync/pipe-barrier.md",
    "docs/isa/scalar/ops/pipeline-sync/get-buf.md",
    "docs/isa/scalar/ops/pipeline-sync/rls-buf.md",
    "docs/isa/scalar/ops/pipeline-sync/mem-bar.md",
    "docs/isa/scalar/ops/pipeline-sync/set-cross-core.md",
    "docs/isa/scalar/ops/pipeline-sync/wait-flag-dev.md",
    "docs/isa/scalar/ops/pipeline-sync/set-intra-block.md",
    "docs/isa/scalar/ops/pipeline-sync/wait-intra-core.md",
    "docs/isa/scalar/dma-copy.md",
    "docs/isa/scalar/ops/dma-copy/set-loop-size-outtoub.md",
    "docs/isa/scalar/ops/dma-copy/set-loop2-stride-outtoub.md",
    "docs/isa/scalar/ops/dma-copy/set-loop1-stride-outtoub.md",
    "docs/isa/scalar/ops/dma-copy/set-loop-size-ubtoout.md",
    "docs/isa/scalar/ops/dma-copy/set-loop2-stride-ubtoout.md",
    "docs/isa/scalar/ops/dma-copy/set-loop1-stride-ubtoout.md",
    "docs/isa/scalar/ops/dma-copy/copy-gm-to-ubuf.md",
    "docs/isa/scalar/ops/dma-copy/copy-ubuf-to-gm.md",
    "docs/isa/scalar/ops/dma-copy/copy-ubuf-to-ubuf.md",
    "docs/isa/scalar/predicate-load-store.md",
    "docs/isa/scalar/ops/predicate-load-store/plds.md",
    "docs/isa/scalar/ops/predicate-load-store/pld.md",
    "docs/isa/scalar/ops/predicate-load-store/pldi.md",
    "docs/isa/scalar/ops/predicate-load-store/psts.md",
    "docs/isa/scalar/ops/predicate-load-store/pst.md",
    "docs/isa/scalar/ops/predicate-load-store/psti.md",
    "docs/isa/scalar/ops/predicate-load-store/pstu.md",
    "docs/isa/scalar/predicate-generation-and-algebra.md",
    "docs/isa/scalar/ops/predicate-generation-and-algebra/pset-b8.md",
    "docs/isa/scalar/ops/predicate-generation-and-algebra/pset-b16.md",
    "docs/isa/scalar/ops/predicate-generation-and-algebra/pset-b32.md",
    "docs/isa/scalar/ops/predicate-generation-and-algebra/pge-b8.md",
    "docs/isa/scalar/ops/predicate-generation-and-algebra/pge-b16.md",
    "docs/isa/scalar/ops/predicate-generation-and-algebra/pge-b32.md",
    "docs/isa/scalar/ops/predicate-generation-and-algebra/plt-b8.md",
    "docs/isa/scalar/ops/predicate-generation-and-algebra/plt-b16.md",
    "docs/isa/scalar/ops/predicate-generation-and-algebra/plt-b32.md",
    "docs/isa/scalar/ops/predicate-generation-and-algebra/ppack.md",
    "docs/isa/scalar/ops/predicate-generation-and-algebra/punpack.md",
    "docs/isa/scalar/ops/predicate-generation-and-algebra/pand.md",
    "docs/isa/scalar/ops/predicate-generation-and-algebra/por.md",
    "docs/isa/scalar/ops/predicate-generation-and-algebra/pxor.md",
    "docs/isa/scalar/ops/predicate-generation-and-algebra/pnot.md",
    "docs/isa/scalar/ops/predicate-generation-and-algebra/psel.md",
    "docs/isa/scalar/ops/predicate-generation-and-algebra/pdintlv-b8.md",
    "docs/isa/scalar/ops/predicate-generation-and-algebra/pintlv-b16.md",
    "docs/isa/scalar/shared-arith.md",
    "docs/isa/scalar/shared-scf.md",
]

COMMUNICATION_REFERENCE_PAGES = [
    "docs/isa/comm/README.md",
    "docs/isa/comm/communication-runtime.md",
    "docs/isa/comm/TBROADCAST.md",
    "docs/isa/comm/TGET.md",
    "docs/isa/comm/TGET_ASYNC.md",
    "docs/isa/comm/TGATHER.md",
    "docs/isa/comm/TNOTIFY.md",
    "docs/isa/comm/TPUT.md",
    "docs/isa/comm/TPUT_ASYNC.md",
    "docs/isa/comm/TREDUCE.md",
    "docs/isa/comm/TSCATTER.md",
    "docs/isa/comm/TTEST.md",
    "docs/isa/comm/TWAIT.md",
]

SYSTEM_SCHEDULING_REFERENCE_PAGES = [
    "docs/isa/system/README.md",
    "docs/isa/system/ops/TFREE.md",
    "docs/isa/system/ops/TPOP.md",
    "docs/isa/system/ops/TPUSH.md",
]

REFERENCE_NOTE_PAGES = [
    "docs/isa/reference/README.md",
    "docs/isa/reference/diagnostics-and-illegal-cases.md",
    "docs/isa/reference/glossary.md",
    "docs/isa/reference/portability-and-target-profiles.md",
    "docs/isa/reference/source-of-truth.md",
]

NAV_PAGES_EN = [
    "index.md",
    "docs/getting-started.md",
    "docs/isa/README.md",
    "docs/isa/introduction/what-is-pto-visa.md",
    "docs/isa/introduction/design-goals-and-boundaries.md",
    "docs/isa/programming-model/tiles-and-valid-regions.md",
    "docs/isa/programming-model/globaltensor-and-data-movement.md",
    "docs/isa/programming-model/auto-vs-manual.md",
    "docs/isa/machine-model/execution-agents.md",
    "docs/isa/machine-model/ordering-and-synchronization.md",
    "docs/isa/memory-model/consistency-baseline.md",
    "docs/isa/memory-model/producer-consumer-ordering.md",
    "docs/isa/state-and-types/type-system.md",
    "docs/isa/state-and-types/location-intent-and-legality.md",
    "docs/isa/syntax-and-operands/assembly-model.md",
    "docs/isa/syntax-and-operands/operands-and-attributes.md",
    "docs/isa/conventions.md",
    *TILE_REFERENCE_PAGES,
    *VECTOR_REFERENCE_PAGES,
    *SCALAR_REFERENCE_PAGES,
    *COMMUNICATION_REFERENCE_PAGES,
    *SYSTEM_SCHEDULING_REFERENCE_PAGES,
    *REFERENCE_NOTE_PAGES,
    "docs/coding/README.md",
    "docs/coding/tutorial.md",
    "docs/reference/pto-intrinsics-header.md",
    "docs/README.md",
    "docs/reference/pto-isa-writing-playbook.md",
    "docs/reference/pto-isa-manual-rewrite-plan.md",
    "docs/reference/pto-isa-manual-review-rubric.md",
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
    # README pages: last segment is a known directory name
    last = base.rsplit("/", 1)[-1]
    if last in README_DIRS:
        return en_url.rstrip("/") + "/README_zh/"
    # general page: append _zh
    return base + "_zh/"


def _generate_lang_map(nav_pages: list[str], existing_urls: set[str]) -> dict:
    """Build a mapping dict for use by the language switcher JS.

    Structure::

        {
          "en_to_zh": { "/docs/isa/README/": "/docs/isa/README_zh/", ... },
          "zh_to_en": { "/docs/isa/README_zh/": "/docs/isa/README/", ... },
          "nav": [
            { "en": "/docs/isa/introduction/what-is-pto-visa/",
              "zh": "/docs/isa/introduction/what-is-pto-visa_zh/",
              "prev_en": "/", "prev_zh": "/index_zh/" },
            ...
          ]
        }
    """
    en_urls = [_md_to_url(p) for p in nav_pages]
    en_to_zh: dict[str, str] = {}
    zh_to_en: dict[str, str] = {}

    for en in en_urls:
        zh = _en_url_to_zh_url(en)
        if zh and zh in existing_urls:
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


def _zh_md_path(rel: str) -> str:
    p = Path(rel)
    if p.name == "README.md":
        return p.with_name("README_zh.md").as_posix()
    if p.name == "index.md":
        return p.with_name("index_zh.md").as_posix()
    return p.with_name(f"{p.stem}_zh.md").as_posix()


def _rel_md_link(from_rel: str, to_rel: str) -> str:
    base = Path(from_rel).parent.as_posix()
    base = "." if base in ("", ".") else base
    return posixpath.relpath(to_rel, base)


def _existing_root_zh_doc_for(rel: str) -> str | None:
    if not rel.startswith("docs/isa/"):
        return None

    stem = Path(rel).stem
    root_name = stem.replace("-", "_").upper() + "_zh.md"

    candidates = [f"docs/isa/{root_name}", f"docs/isa/comm/{root_name}"]
    for candidate in candidates:
        if (REPO_ROOT / candidate).exists():
            return candidate
    return None


def _zh_context_links_for(rel: str) -> list[tuple[str, str]]:
    links: list[tuple[str, str]] = []
    if rel.startswith("docs/isa/introduction/"):
        links.append(("中文 ISA 入口", "docs/isa/README_zh.md"))
    elif rel.startswith("docs/isa/programming-model/"):
        links.append(("中文编程模型入口", "docs/isa/programming-model/README_zh.md"))
    elif rel.startswith("docs/isa/machine-model/"):
        links.append(("中文机器模型入口", "docs/isa/machine-model/README_zh.md"))
    elif rel.startswith("docs/isa/memory-model/"):
        links.append(("中文内存模型入口", "docs/isa/memory-model/README_zh.md"))
    elif rel.startswith("docs/isa/state-and-types/"):
        links.append(("中文状态与类型入口", "docs/isa/state-and-types/README_zh.md"))
    elif rel.startswith("docs/isa/syntax-and-operands/"):
        links.append(("中文语法与操作数入口", "docs/isa/syntax-and-operands/README_zh.md"))
    elif rel.startswith("docs/isa/instruction-families/"):
        links.append(("中文 ISA 指令参考入口", "docs/isa/README_zh.md"))
    elif (
        rel.startswith("docs/isa/tile/")
        or rel.startswith("docs/isa/vector/")
        or rel.startswith("docs/isa/scalar/")
        or rel.startswith("docs/isa/comm/")
        or rel.startswith("docs/isa/system/")
    ):
        links.append(("中文 ISA 指令参考入口", "docs/isa/README_zh.md"))
    return links


def _write_missing_zh_wrapper(en_rel: str, zh_rel: str) -> None:
    title = _extract_first_heading(REPO_ROOT / en_rel)
    with mkdocs_gen_files.open(zh_rel, "w") as f:
        f.write(f"# {title}\n\n")
        f.write("自动生成的中文入口页，用于保证中文导航保持在中文路径下。\n\n")
        f.write("## 当前状态\n\n")
        f.write(
            f"- [对应英文页面]({_rel_md_link(zh_rel, en_rel)})\n"
            f"- [中文手册入口]({_rel_md_link(zh_rel, 'docs/PTO-Virtual-ISA-Manual_zh.md')})\n"
        )

        existing_zh = _existing_root_zh_doc_for(en_rel)
        if existing_zh:
            f.write(f"- [现有中文指令说明]({_rel_md_link(zh_rel, existing_zh)})\n")

        extra_links = _zh_context_links_for(en_rel)
        if extra_links:
            for label, target in extra_links:
                f.write(f"- [{label}]({_rel_md_link(zh_rel, target)})\n")

        f.write("\n## 说明\n\n")
        f.write(
            "当前 PTO ISA 的新英文手册结构已经展开，但对应的中文正文尚未完全按新结构补齐。"
            "在中文导航中点击本页时，你仍然停留在中文路径下；若需要完整细节，请使用上面的英文页面或已有中文参考页。\n"
        )


def _generate_missing_zh_wrappers(copied_md: list[str]) -> list[str]:
    generated: list[str] = []
    existing = set(copied_md)

    for en_rel in NAV_PAGES_EN:
        if en_rel.endswith("_zh.md"):
            continue
        zh_rel = _zh_md_path(en_rel)
        if zh_rel in existing:
            continue
        if not (REPO_ROOT / en_rel).exists():
            continue
        _write_missing_zh_wrapper(en_rel, zh_rel)
        generated.append(zh_rel)
        existing.add(zh_rel)

    return generated


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    copied_md: list[str] = []

    mkdocs_src = REPO_ROOT / "docs" / "mkdocs" / "src"

    # Remove stale generated mirrors from previous builds. These paths are ignored
    # build artifacts; authored sources must live in docs/isa/ or the small
    # MkDocs-owned allowlist below.
    for stale_root in GENERATED_MKDOCS_ROOTS:
        stale_path = mkdocs_src / stale_root
        if stale_path.exists():
            shutil.rmtree(stale_path)
    for stale_root_file in GENERATED_MKDOCS_FILES:
        stale_path = mkdocs_src / stale_root_file
        if stale_path.exists():
            stale_path.unlink()

    # Step 1: Process the small MkDocs-owned Markdown allowlist.
    for virtual_path in sorted(HANDWRITTEN_MKDOCS_MD):
        src = mkdocs_src / virtual_path
        if not src.exists():
            continue
        text = src.read_text(encoding="utf-8-sig", errors="replace")
        text = _strip_generated_header(text)
        text = _rewrite_rel_imgs_for_build(text, virtual_path)
        text = _rewrite_links_for_build(text, virtual_path)
        with mkdocs_gen_files.open(virtual_path, "w") as f:
            f.write(text)
        copied_md.append(virtual_path)

    # Step 2: Publish curated canonical repo markdown preserving its real path.
    for src in REPO_ROOT.rglob("*.md"):
        rel = src.relative_to(REPO_ROOT).as_posix()
        if _should_skip(rel):
            continue
        if not _is_published_md(rel):
            continue
        # Use utf-8-sig to automatically remove BOM if present
        text = src.read_text(encoding="utf-8-sig", errors="replace")
        # Rewrite relative <img src="..."> paths for published repo files.
        text = _rewrite_rel_imgs_for_build(text, rel)
        # Rewrite mkdocs/src/... links to root-absolute /... links, then
        # let _rewrite_links_for_build convert them to correct relative paths.
        if MKDOCS_SRC_LINK_RE.search(text):
            text = MKDOCS_SRC_LINK_RE.sub(r"](/\1)", text)
            text = _rewrite_links_for_build(text, rel)
        with mkdocs_gen_files.open(rel, "w") as f:
            f.write(f"<!-- Generated from `{rel}` -->\n\n")
            f.write(text)
        copied_md.append(rel)

    # Step 3: For English-first manual pages that do not yet have authored
    # Chinese counterparts, generate Chinese wrapper pages so the Chinese
    # sidebar stays on Chinese URLs instead of falling back to English paths.
    copied_md.extend(_generate_missing_zh_wrappers(copied_md))

    all_md = sorted(set(copied_md))
    published_md = [rel for rel in all_md if _is_published_md(rel)]

    # Generate lang-map.json for zero-latency language switching.
    existing_urls = {_md_to_url(p) for p in published_md}
    lang_map = _generate_lang_map(NAV_PAGES_EN, existing_urls)
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
        if not _is_published_asset(rel):
            continue
        with mkdocs_gen_files.open(rel, "wb") as f:
            f.write(src.read_bytes())


main()
