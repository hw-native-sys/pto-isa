// --------------------------------------------------------------------------------
// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// --------------------------------------------------------------------------------

/**
 * nav-translator.js
 *
 * Two responsibilities:
 *  1. Translate sidebar navigation text labels EN <-> ZH.
 *  2. Rewrite nav link hrefs to the correct language version using
 *     the pre-built lang-map.json lookup table (no heuristics).
 *
 * On a ZH page the links are rewritten immediately on DOMContentLoaded.
 * When the user switches language via language-switcher.js the public
 * window.translateNavigation / window.restoreEnglishNavigation hooks are
 * called before the page redirect.
 */
(function () {
    'use strict';

    // ── translation table ────────────────────────────────────────────────────

    var NAV_TRANSLATIONS = {
        'Home': '\u9996\u9875',
        'Getting Started': '\u5feb\u901f\u5f00\u59cb',
        'PTO Virtual ISA Manual': 'PTO \u865a\u62df ISA \u624b\u518c',
        'Programming Model': '\u7f16\u7a0b\u6a21\u578b',
        'ISA Reference': 'ISA \u53c2\u8003',
        'Machine Model': '\u673a\u5668\u6a21\u578b',
        'Examples': '\u793a\u4f8b',
        'Examples & Kernels': '\u793a\u4f8b\u4e0e\u7b97\u5b50',
        'Documentation': '\u6587\u6863',
        'Full Index': '\u5b8c\u6574\u7d22\u5f15',
        'Preface': '\u524d\u8a00',
        'Overview': '\u6982\u8ff0',
        'Execution Model': '\u6267\u884c\u6a21\u578b',
        'State and Types': '\u72b6\u6001\u4e0e\u7c7b\u578b',
        'Tiles and GlobalTensor': 'Tile \u4e0e GlobalTensor',
        'Synchronization': '\u540c\u6b65',
        'PTO Assembly (PTO-AS)': 'PTO \u6c47\u7f16 (PTO-AS)',
        'Instruction Set (overview)': '\u6307\u4ee4\u96c6\uff08\u6982\u8ff0\uff09',
        'Programming Guide': '\u7f16\u7a0b\u6307\u5357',
        'Virtual ISA and IR': '\u865a\u62df ISA \u4e0e IR',
        'Bytecode and Toolchain': '\u5b57\u8282\u7801\u4e0e\u5de5\u5177\u94fe',
        'Memory Ordering and Consistency': '\u5185\u5b58\u987a\u5e8f\u4e0e\u4e00\u81f4\u6027',
        'Backend Profiles and Conformance': '\u540e\u7aef\u914d\u7f6e\u4e0e\u4e00\u81f4\u6027',
        'Glossary': '\u672f\u8bed\u8868',
        'Instruction Contract Template': '\u6307\u4ee4\u5951\u7ea6\u6a21\u677f',
        'Diagnostics Taxonomy': '\u8bca\u65ad\u5206\u7c7b',
        'Instruction Family Matrix': '\u6307\u4ee4\u65cf\u77e9\u9635',
        'Tile': 'Tile',
        'GlobalTensor': 'GlobalTensor',
        'Scalar': 'Scalar',
        'Event': 'Event',
        'Tutorial': '\u6559\u7a0b',
        'Tutorials': '\u6559\u7a0b\u96c6',
        'Vec Add': '\u5411\u91cf\u52a0\u6cd5',
        'Row Softmax': '\u884c Softmax',
        'GEMM': 'GEMM',
        'Optimization': '\u4f18\u5316',
        'Debugging': '\u8c03\u8bd5',
        'ISA Conventions': 'ISA \u7ea6\u5b9a',
        'PTO AS Reference': 'PTO AS \u53c2\u8003',
        'PTO-AS Specification': 'PTO-AS \u89c4\u8303',
        'Conventions': '\u7ea6\u5b9a',
        'Non-ISA Operations': '\u975e ISA \u64cd\u4f5c',
        'Elementwise Operations': '\u9010\u5143\u7d20\u64cd\u4f5c',
        'Tile-Scalar Operations': 'Tile-\u6807\u91cf\u64cd\u4f5c',
        'Axis Operations': '\u8f74\u64cd\u4f5c',
        'Memory Operations': '\u5185\u5b58\u64cd\u4f5c',
        'Matrix Operations': '\u77e9\u9635\u64cd\u4f5c',
        'Data Movement Operations': '\u6570\u636e\u642c\u8fd0\u64cd\u4f5c',
        'Complex Operations': '\u590d\u6742\u64cd\u4f5c',
        'Manual Binding Operations': '\u624b\u52a8\u7ed1\u5b9a\u64cd\u4f5c',
        'Scalar Arithmetic Operations': '\u6807\u91cf\u7b97\u672f\u64cd\u4f5c',
        'Control Flow Operations': '\u63a7\u5236\u6d41\u64cd\u4f5c',
        'PTO ISA Table': 'PTO ISA \u8868',
        'Manual / Resource Binding': '\u624b\u52a8/\u8d44\u6e90\u7ed1\u5b9a',
        'Elementwise (Tile-Tile)': '\u9010\u5143\u7d20\uff08Tile-Tile\uff09',
        'Tile-Scalar / Tile-Immediate': 'Tile-\u6807\u91cf/Tile-\u7acb\u5373\u6570',
        'Axis Reduce / Expand': '\u8f74\u5f52\u7ea6/\u6269\u5c55',
        'Memory (GM <-> Tile)': '\u5185\u5b58\uff08GM <-> Tile\uff09',
        'Matrix Multiply': '\u77e9\u9635\u4e58',
        'Data Movement / Layout': '\u6570\u636e\u642c\u8fd0/\u5e03\u5c40',
        'Complex Instructions': '\u590d\u6742\u6307\u4ee4',
        'Communication': '\u901a\u4fe1',
        'TGATHER (comm)': 'TGATHER (\u901a\u4fe1)',
        'TSCATTER (comm)': 'TSCATTER (\u901a\u4fe1)',
        'Reference': '\u53c2\u8003',
        'Intrinsics Header': '\u5185\u5efa\u51fd\u6570\u5934\u6587\u4ef6',
        'All Instructions Index': '\u5168\u90e8\u6307\u4ee4\u7d22\u5f15',
        'Abstract Machine': '\u62bd\u8c61\u673a\u5668',
        'Machine Index': '\u673a\u5668\u7d22\u5f15',
        'High-Performance Kernels': '\u9ad8\u6027\u80fd\u7b97\u5b50',
        'GEMM Performance': 'GEMM \u6027\u80fd',
        'Flash Attention': 'Flash Attention',
        'Baseline Demos': '\u57fa\u7840\u793a\u4f8b',
        'Add Demo': '\u52a0\u6cd5\u793a\u4f8b',
        'GEMM Demo': 'GEMM \u793a\u4f8b',
        'Tests': '\u6d4b\u8bd5',
        'Tests Overview': '\u6d4b\u8bd5\u6982\u89c8',
        'Test Scripts': '\u6d4b\u8bd5\u811a\u672c',
        'Docs Index': '\u6587\u6863\u7d22\u5f15',
        'Build Documentation': '\u6784\u5efa\u6587\u6863',
    };

    // ── helpers ───────────────────────────────────────────────────────────────

    function getCurrentLanguage() {
        var p = window.location.pathname;
        return (p.indexOf('_zh/') !== -1 || p.slice(-8) === '_zh.html') ? 'zh' : 'en';
    }

    /**
     * Resolve a (possibly relative) href to an absolute pathname with
     * a trailing slash, matching the keys in lang-map.json.
     */
    function resolveToAbsPath(href) {
        if (!href || href.charAt(0) === '#') return null;
        try {
            var abs = new URL(href, window.location.href).pathname;
            if (abs.charAt(abs.length - 1) !== '/') abs += '/';
            return abs;
        } catch (_) {
            return null;
        }
    }

    // ── link rewriting ────────────────────────────────────────────────────────

    function rewriteNavLinksToZh(enToZh) {
        var links = document.querySelectorAll(
            '.wy-menu-vertical a, nav a, .toctree-l1 > a, .toctree-l2 > a'
        );
        links.forEach(function (link) {
            if (!link.hasAttribute('data-original-href')) {
                link.setAttribute('data-original-href', link.getAttribute('href') || '');
            }
            var origHref = link.getAttribute('data-original-href');
            if (!origHref || origHref.charAt(0) === '#' ||
                origHref.indexOf('http') === 0) return;

            var absPath = resolveToAbsPath(origHref);
            if (!absPath) return;

            var zhHref = enToZh[absPath];
            if (zhHref) link.setAttribute('href', zhHref);
        });
    }

    function restoreNavLinksToEn() {
        var links = document.querySelectorAll(
            '.wy-menu-vertical a, nav a, .toctree-l1 > a, .toctree-l2 > a'
        );
        links.forEach(function (link) {
            var orig = link.getAttribute('data-original-href');
            if (orig != null) link.setAttribute('href', orig);
        });
    }

    // ── text translation ──────────────────────────────────────────────────────

    function translateTextLabels() {
        var links = document.querySelectorAll(
            '.wy-menu-vertical a, nav a, .toctree-l1 > a, .toctree-l2 > a'
        );
        links.forEach(function (link) {
            var orig = link.textContent.trim();
            if (!orig) return;
            if (!link.hasAttribute('data-original-text')) {
                link.setAttribute('data-original-text', orig);
            }
            var key = link.getAttribute('data-original-text');
            if (NAV_TRANSLATIONS[key]) link.textContent = NAV_TRANSLATIONS[key];
        });

        var captions = document.querySelectorAll('.caption-text');
        captions.forEach(function (caption) {
            var orig = caption.textContent.trim();
            if (!orig) return;
            if (!caption.hasAttribute('data-original-text')) {
                caption.setAttribute('data-original-text', orig);
            }
            var key = caption.getAttribute('data-original-text');
            if (NAV_TRANSLATIONS[key]) caption.textContent = NAV_TRANSLATIONS[key];

            var parent = caption.closest('p.caption');
            if (parent && !parent.hasAttribute('data-click-protected')) {
                parent.setAttribute('data-click-protected', 'true');
                parent.addEventListener('click', function (e) {
                    if (e.target === caption || e.target === parent ||
                        e.target.classList.contains('caption-text')) {
                        e.preventDefault();
                        e.stopPropagation();
                    }
                }, true);
            }
        });

        var siteTitle = document.querySelector('.wy-side-nav-search a, .navbar-brand');
        if (siteTitle) {
            if (!siteTitle.hasAttribute('data-original-title')) {
                siteTitle.setAttribute('data-original-title', siteTitle.textContent);
            }
            if ((siteTitle.getAttribute('data-original-title') || '').indexOf('PTO Virtual ISA') !== -1) {
                siteTitle.textContent = 'PTO \u865a\u62df ISA \u67b6\u6784\u624b\u518c';
            }
        }
    }

    function restoreTextLabels() {
        var links = document.querySelectorAll(
            '.wy-menu-vertical a, nav a, .toctree-l1 > a, .toctree-l2 > a'
        );
        links.forEach(function (link) {
            var orig = link.getAttribute('data-original-text');
            if (orig) link.textContent = orig;
        });

        var captions = document.querySelectorAll('.caption-text');
        captions.forEach(function (caption) {
            var orig = caption.getAttribute('data-original-text');
            if (orig) caption.textContent = orig;
        });

        var siteTitle = document.querySelector('.wy-side-nav-search a, .navbar-brand');
        if (siteTitle) {
            var orig = siteTitle.getAttribute('data-original-title');
            if (orig) siteTitle.textContent = orig;
        }
    }

    // ── auto-apply on ZH pages ────────────────────────────────────────────────

    /**
     * On a Chinese page, rewrite nav links immediately using the cached map
     * so every sidebar link points to the correct ZH page.
     */
    function autoApplyOnZhPage() {
        if (getCurrentLanguage() !== 'zh') return;
        var loader = window.loadLangMap ? window.loadLangMap() : Promise.resolve(null);
        loader.then(function (map) {
            if (!map) return;
            translateTextLabels();
            rewriteNavLinksToZh(map.en_to_zh);
            // Re-apply after a short delay to catch any links rendered late by the theme.
            setTimeout(function () { rewriteNavLinksToZh(map.en_to_zh); }, 300);
        });
    }

    function init() {
        // Auto-apply on ZH pages: rewrite nav links and translate labels.
        autoApplyOnZhPage();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    // ── public API ────────────────────────────────────────────────────────────

    window.translateNavigation = function (targetLang) {
        if (targetLang !== 'zh') return;
        var loader = window.loadLangMap ? window.loadLangMap() : Promise.resolve(null);
        loader.then(function (map) {
            translateTextLabels();
            if (map) rewriteNavLinksToZh(map.en_to_zh);
        });
    };

    window.restoreEnglishNavigation = function () {
        restoreTextLabels();
        restoreNavLinksToEn();
    };

})();
 