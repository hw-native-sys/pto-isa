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
 * Language switcher — icon + dropdown.
 *
 * Languages are treated as separate reading tracks. Switching language should
 * move the user to a real counterpart page when one exists; otherwise it
 * should fall back to a language landing page instead of fabricating URLs or
 * trying to translate the current page in place.
 */
(function () {
    'use strict';

    const LANG_KEY = 'preferred_language';
    const MAP_CACHE_KEY = 'lang_map_cache_v2';
    const LANG_MAP_URL = '/lang-map.json';
    const LANGUAGE_HOME = {
        en: '/',
        zh: '/index_zh/',
    };
    const LANGUAGE_DOC_HOME = {
        en: '/docs/isa/',
        zh: '/docs/isa/README_zh/',
    };

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

    function getLanguageHome(targetLang) {
        const p = window.location.pathname;
        if (p.startsWith('/docs/')) {
            return LANGUAGE_DOC_HOME[targetLang];
        }
        return LANGUAGE_HOME[targetLang];
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

    function getTargetUrl(map, targetLang) {
        return getAlternateUrl(map, targetLang) || getLanguageHome(targetLang);
    }

    // ── UI ───────────────────────────────────────────────────────────────────

    function createSwitcher() {
        const currentLang = getCurrentLanguage();
        const container = document.createElement('div');
        container.id = 'language-switcher';
        container.className = 'language-switcher';

        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'language-switcher__button';
        button.setAttribute('aria-haspopup', 'true');
        button.setAttribute('aria-expanded', 'false');
        button.title = 'Switch language';
        button.innerHTML = '<span class="language-switcher__icon">🌐</span><span class="language-switcher__label">' +
            (currentLang === 'zh' ? '中文' : 'EN') + '</span>';

        const menu = document.createElement('div');
        menu.className = 'language-switcher__menu';
        menu.hidden = true;

        const langs = [
            { code: 'en', label: 'English' },
            { code: 'zh', label: '中文' },
        ];

        langs.forEach(function (lang) {
            const item = document.createElement('a');
            item.href = '#';
            item.className = 'language-switcher__item';
            if (lang.code === currentLang) {
                item.classList.add('is-active');
            }
            item.dataset.langCode = lang.code;
            item.textContent = lang.label;
            item.addEventListener('click', function (e) {
                e.preventDefault();
                const targetLang = lang.code;
                setPreferredLanguage(targetLang);
                loadLangMap().then(function (map) {
                    const url = getTargetUrl(map, targetLang);
                    if (url !== window.location.pathname) {
                        window.location.href = url;
                    } else {
                        menu.hidden = true;
                        button.setAttribute('aria-expanded', 'false');
                    }
                });
            });
            menu.appendChild(item);
        });

        button.addEventListener('click', function () {
            const expanded = button.getAttribute('aria-expanded') === 'true';
            button.setAttribute('aria-expanded', expanded ? 'false' : 'true');
            menu.hidden = expanded;
        });

        document.addEventListener('click', function (e) {
            if (!container.contains(e.target)) {
                menu.hidden = true;
                button.setAttribute('aria-expanded', 'false');
            }
        });

        container.appendChild(button);
        container.appendChild(menu);
        document.body.appendChild(container);
    }

    // ── init ─────────────────────────────────────────────────────────────────

    function init() {
        createSwitcher();
        loadLangMap();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
