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
 * PTO site chrome helpers:
 * - mount the language switcher inside the mobile header so it is always visible
 * - add a compact manual utility bar on deep pages with home/reference/search
 */
(function () {
    'use strict';

    function getCurrentLanguage() {
        var p = window.location.pathname;
        return (p.indexOf('_zh/') !== -1 || p.slice(-8) === '_zh.html') ? 'zh' : 'en';
    }

    function isHomePage() {
        return window.location.pathname === '/' || window.location.pathname === '/index_zh/';
    }

    function isSearchPage() {
        return window.location.pathname === '/search.html';
    }

    function getLabels(lang) {
        if (lang === 'zh') {
            return {
                home: '中文首页',
                reference: 'ISA 参考',
                search: '搜索手册',
                placeholder: '搜索 PTO ISA',
                button: '搜索',
                homeHref: '/index_zh/',
                referenceHref: '/docs/isa/README_zh/',
            };
        }

        return {
            home: 'Manual Home',
            reference: 'ISA Index',
            search: 'Search Manual',
            placeholder: 'Search PTO ISA',
            button: 'Search',
            homeHref: '/',
            referenceHref: '/docs/isa/',
        };
    }

    function rewriteMarkdownHref(href) {
        if (!href || href.charAt(0) === '#') {
            return null;
        }

        try {
            var url = new URL(href, window.location.href);
            if (url.origin !== window.location.origin || !url.pathname.endsWith('.md')) {
                return null;
            }

            var pathname = url.pathname;
            if (pathname.endsWith('/README.md')) {
                pathname = pathname.slice(0, -'README.md'.length);
            } else if (pathname.endsWith('/index.md')) {
                pathname = pathname.slice(0, -'index.md'.length);
            } else {
                pathname = pathname.slice(0, -3) + '/';
            }

            pathname = pathname.replace(/\/+/g, '/');
            if (pathname === '') {
                pathname = '/';
            }

            return pathname + url.search + url.hash;
        } catch (_) {
            return null;
        }
    }

    function normalizeMarkdownLinks() {
        document.querySelectorAll('a[href$=".md"]').forEach(function (link) {
            var rewritten = rewriteMarkdownHref(link.getAttribute('href'));
            if (!rewritten) {
                return;
            }

            link.setAttribute('href', rewritten);
            link.classList.add('reference', 'internal');
        });
    }

    function mountLanguageSwitcher() {
        var switcher = document.getElementById('language-switcher');
        if (!switcher) {
            return;
        }

        var topNav = document.querySelector('.wy-nav-top');
        var useMobileHeader = window.matchMedia('(max-width: 768px)').matches && topNav;
        var desiredParent = useMobileHeader ? topNav : document.body;

        if (switcher.parentElement !== desiredParent) {
            desiredParent.appendChild(switcher);
        }

        switcher.classList.toggle('language-switcher--in-header', !!useMobileHeader);
        document.body.classList.toggle('pto-mobile-header-active', !!useMobileHeader);
    }

    function buildUtilityBar() {
        if (isHomePage() || isSearchPage()) {
            return;
        }

        var content = document.querySelector('.wy-nav-content');
        var breadcrumbs = document.querySelector('.wy-breadcrumbs');
        if (!content || !breadcrumbs || content.querySelector('.manual-utility-bar')) {
            return;
        }

        var lang = getCurrentLanguage();
        var labels = getLabels(lang);
        var bar = document.createElement('div');
        bar.className = 'manual-utility-bar';

        var links = document.createElement('div');
        links.className = 'manual-utility-links';

        [
            { href: labels.homeHref, text: labels.home },
            { href: labels.referenceHref, text: labels.reference },
            { href: '/search.html', text: labels.search },
        ].forEach(function (item) {
            var link = document.createElement('a');
            link.href = item.href;
            link.className = 'manual-utility-link';
            link.textContent = item.text;
            links.appendChild(link);
        });

        var form = document.createElement('form');
        form.className = 'manual-search-form';
        form.action = '/search.html';
        form.method = 'get';

        var input = document.createElement('input');
        input.type = 'search';
        input.name = 'q';
        input.placeholder = labels.placeholder;
        input.setAttribute('aria-label', labels.placeholder);

        var button = document.createElement('button');
        button.type = 'submit';
        button.textContent = labels.button;

        form.appendChild(input);
        form.appendChild(button);
        bar.appendChild(links);
        bar.appendChild(form);

        breadcrumbs.insertAdjacentElement('afterend', bar);
    }

    function init() {
        window.requestAnimationFrame(function () {
            normalizeMarkdownLinks();
            mountLanguageSwitcher();
            buildUtilityBar();
        });

        window.addEventListener('resize', function () {
            mountLanguageSwitcher();
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
