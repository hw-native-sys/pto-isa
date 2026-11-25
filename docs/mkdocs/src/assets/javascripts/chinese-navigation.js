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
 * Chinese page prev/next navigation — static-map edition.
 *
 * Replaces the old implementation that fetched the corresponding English page
 * at runtime.  Instead we read the nav order from the pre-built lang-map.json
 * (already cached in sessionStorage by language-switcher.js) and build the
 * buttons synchronously — zero additional network requests.
 */
(function () {
    'use strict';

    function getCurrentLanguage() {
        const p = window.location.pathname;
        return (p.includes('_zh/') || p.endsWith('_zh.html')) ? 'zh' : 'en';
    }

    function findNavEntry(map, pathname) {
        // Try direct match in nav array.
        for (const entry of map.nav) {
            if (entry.zh === pathname) return entry;
        }
        return null;
    }

    // ── DOM helpers ──────────────────────────────────────────────────────────

    function createNavButtons(prevPage, nextPage) {
        let footer = document.querySelector('footer');
        if (!footer) {
            const main = document.querySelector('.wy-nav-content');
            if (!main) return;
            footer = document.createElement('footer');
            main.appendChild(footer);
        }

        // Remove any pre-existing (empty) nav container.
        const existing = footer.querySelector('.rst-footer-buttons');
        if (existing) existing.remove();

        const nav = document.createElement('div');
        nav.className = 'rst-footer-buttons';
        nav.setAttribute('role', 'navigation');
        nav.setAttribute('aria-label', 'Footer Navigation');

        if (prevPage) {
            const btn = document.createElement('a');
            btn.href = prevPage.href;
            btn.className = 'btn btn-neutral float-left';
            btn.title = prevPage.title || '';
            btn.innerHTML = '<span class="icon icon-circle-arrow-left"></span> \u4E0A\u4E00\u9875';
            nav.appendChild(btn);
        }

        if (nextPage) {
            const btn = document.createElement('a');
            btn.href = nextPage.href;
            btn.className = 'btn btn-neutral float-right';
            btn.title = nextPage.title || '';
            btn.innerHTML = '\u4E0B\u4E00\u9875 <span class="icon icon-circle-arrow-right"></span>';
            nav.appendChild(btn);
        }

        footer.insertBefore(nav, footer.firstChild);

        // Also update the rst-versions bar if present.
        const rstCur = document.querySelector('.rst-versions .rst-current-version');
        if (rstCur) {
            rstCur.innerHTML = '';
            if (prevPage) {
                const s = document.createElement('span');
                const a = document.createElement('a');
                a.href = prevPage.href;
                a.style.color = '#fcfcfc';
                a.textContent = '\u00AB \u4E0A\u4E00\u9875';
                s.appendChild(a);
                rstCur.appendChild(s);
            }
            if (nextPage) {
                const s = document.createElement('span');
                s.style.float = 'right';
                const a = document.createElement('a');
                a.href = nextPage.href;
                a.style.color = '#fcfcfc';
                a.textContent = '\u4E0B\u4E00\u9875 \u00BB';
                s.appendChild(a);
                rstCur.appendChild(s);
            }
        }
    }

    // ── main ─────────────────────────────────────────────────────────────────

    function generateNav() {
        if (getCurrentLanguage() !== 'zh') return;

        // Reuse the map already fetched/cached by language-switcher.js.
        const loader = window.loadLangMap ? window.loadLangMap() : Promise.resolve(null);

        loader.then(map => {
            if (!map) return;

            const entry = findNavEntry(map, window.location.pathname);
            if (!entry) return;

            const prevPage = entry.prev_zh ? { href: entry.prev_zh } : null;
            const nextPage = entry.next_zh ? { href: entry.next_zh } : null;

            createNavButtons(prevPage, nextPage);
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', generateNav);
    } else {
        generateNav();
    }
})();
