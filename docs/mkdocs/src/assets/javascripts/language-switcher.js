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
 * Language switcher — static-map edition.
 *
 * At build time gen_pages.py emits /lang-map.json which contains the complete
 * en⇔zh URL mapping for every page in the nav.  This module fetches that map
 * once (cached in sessionStorage), then switches pages with a direct lookup —
 * no heuristics, no per-page fetch, no MutationObserver overhead.
 */
(function () {
    'use strict';

    const LANG_KEY = 'preferred_language';
    const MAP_CACHE_KEY = 'lang_map_cache_v2';
    const LANG_MAP_URL = '/lang-map.json';

    // ── helpers ──────────────────────────────────────────────────────────────

    function getPreferredLanguage() {
        return localStorage.getItem(LANG_KEY) || 'en';
    }

    function setPreferredLanguage(lang) {
        localStorage.setItem(LANG_KEY, lang);
    }

    function getCurrentLanguage() {
        const p = window.location.pathname;
        return (p.includes('_zh/') || p.endsWith('_zh.html')) ? 'zh' : 'en';
    }

    // ── map loading (fetch once, cache in sessionStorage) ────────────────────

    let _mapPromise = null;

    function loadLangMap() {
        if (_mapPromise) return _mapPromise;

        // Try sessionStorage first to avoid repeated fetches within a session.
        try {
            const cached = sessionStorage.getItem(MAP_CACHE_KEY);
            if (cached) {
                const map = JSON.parse(cached);
                _mapPromise = Promise.resolve(map);
                return _mapPromise;
            }
        } catch (_) { /* ignore */ }

        _mapPromise = fetch(LANG_MAP_URL)
            .then(r => {
                if (!r.ok) throw new Error('lang-map.json not found');
                return r.json();
            })
            .then(map => {
                try { sessionStorage.setItem(MAP_CACHE_KEY, JSON.stringify(map)); } catch (_) { }
                return map;
            })
            .catch(err => {
                console.warn('[lang-switcher] Could not load lang-map.json:', err);
                _mapPromise = null;
                return null;
            });

        return _mapPromise;
    }

    // Expose for chinese-navigation.js
    window.loadLangMap = loadLangMap;

    // ── URL lookup ───────────────────────────────────────────────────────────

    function getAlternateUrl(map, targetLang) {
        if (!map) return null;
        const cur = window.location.pathname;
        if (targetLang === 'zh') {
            return map.en_to_zh[cur] || null;
        } else {
            return map.zh_to_en[cur] || null;
        }
    }

    // ── UI ───────────────────────────────────────────────────────────────────

    function createSwitcher(preferredLang) {
        const container = document.createElement('div');
        container.id = 'language-switcher';
        container.style.cssText = [
            'position:fixed', 'top:10px', 'right:10px', 'z-index:1000',
            'background:#fff', 'border:1px solid #ccc', 'border-radius:4px',
            'padding:5px 10px', 'box-shadow:0 2px 4px rgba(0,0,0,.1)',
            'font-family:sans-serif', 'font-size:14px',
        ].join(';');

        const langs = [
            { code: 'en', label: '\uD83C\uDDEC\uD83C\uDDE7 English' },
            { code: 'zh', label: '\uD83C\uDDE8\uD83C\uDDF3 \u4E2D\u6587' },
        ];

        langs.forEach((lang, i) => {
            if (i > 0) {
                const sep = document.createElement('span');
                sep.textContent = ' | ';
                sep.style.color = '#999';
                container.appendChild(sep);
            }

            const a = document.createElement('a');
            a.href = '#';
            a.textContent = lang.label;
            a.dataset.langCode = lang.code;
            a.style.cssText = [
                'text-decoration:none',
                'cursor:pointer',
                `color:${preferredLang === lang.code ? '#2980b9' : '#333'}`,
                `font-weight:${preferredLang === lang.code ? 'bold' : 'normal'}`,
            ].join(';');

            a.addEventListener('click', function (e) {
                e.preventDefault();
                const targetLang = lang.code;
                setPreferredLanguage(targetLang);

                // Update switcher appearance immediately.
                container.querySelectorAll('a').forEach(el => {
                    const active = el.dataset.langCode === targetLang;
                    el.style.color = active ? '#2980b9' : '#333';
                    el.style.fontWeight = active ? 'bold' : 'normal';
                });

                // Translate nav labels (text only, no link rewrite needed).
                if (targetLang === 'zh' && typeof window.translateNavigation === 'function') {
                    window.translateNavigation('zh');
                } else if (targetLang === 'en' && typeof window.restoreEnglishNavigation === 'function') {
                    window.restoreEnglishNavigation();
                }

                // Navigate to the alternate page via map lookup.
                loadLangMap().then(map => {
                    const url = getAlternateUrl(map, targetLang);
                    if (url && url !== window.location.pathname) {
                        window.location.href = url;
                    }
                    // If not found in map, stay on current page (no broken redirect).
                });
            });

            a.addEventListener('mouseenter', function () {
                if (this.dataset.langCode !== getPreferredLanguage()) {
                    this.style.color = '#2980b9';
                }
            });
            a.addEventListener('mouseleave', function () {
                if (this.dataset.langCode !== getPreferredLanguage()) {
                    this.style.color = '#333';
                }
            });

            container.appendChild(a);
        });

        document.body.appendChild(container);
    }

    // ── init ─────────────────────────────────────────────────────────────────

    function init() {
        const preferred = getPreferredLanguage();
        createSwitcher(preferred);

        // Kick off map pre-fetch so it is cached before the user clicks.
        loadLangMap();

        // Translate nav labels if user prefers Chinese.
        if (preferred === 'zh' && typeof window.translateNavigation === 'function') {
            window.translateNavigation('zh');
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
