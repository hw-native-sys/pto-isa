// --------------------------------------------------------------------------------
// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// --------------------------------------------------------------------------------

// MathJax configuration for MkDocs.
//
// This file must be loaded *before* MathJax itself (see `docs/mkdocs/mkdocs.yml`).
// We rely on `pymdownx.arithmatex` to wrap math, and MathJax renders it.
window.MathJax = {
  tex: {
    inlineMath: [
      ["$", "$"],
      ["\\(", "\\)"],
    ],
    displayMath: [
      ["$$", "$$"],
      ["\\[", "\\]"],
    ],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    // Keep the default tag list, but allow math inside code blocks where arithmatex emits it.
    ignoreHtmlClass: "no-mathjax",
    processHtmlClass: "arithmatex",
  },
};
