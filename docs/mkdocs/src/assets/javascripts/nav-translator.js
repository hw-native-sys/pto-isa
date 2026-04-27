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
        'PTO ISA Manual': 'PTO ISA 手册',
        'PTO Virtual ISA Manual': 'PTO \u865a\u62df ISA \u624b\u518c',
        'Introduction': '引言',
        'What Is PTO VISA?': '什么是 PTO 指令集架构？',
        'What Is PTO ISA?': '什么是 PTO 指令集架构？',
        'Parallel Tile Operation ISA Version 1.0': 'PTO指令集架构 1.0',
        'Goals Of PTO': 'PTO 的目标',
        'Why PTO Exists': '为什么需要 PTO',
        'PTO ISA Version 1.0': 'PTO指令集架构 1.0',
        'Design Goals And Boundaries': '设计目标与边界',
        'Scope And Boundaries': '范围与边界',
        'Programming Model': '\u7f16\u7a0b\u6a21\u578b',
        'Machine Model': '\u673a\u5668\u6a21\u578b',
        'Overview': '\u6982\u8ff0',
        'Tiles And Valid Regions': 'Tile 与有效区域',
        'GlobalTensor And Data Movement': 'GlobalTensor 与数据搬运',
        'Auto Vs Manual': 'Auto 与 Manual',
        'Execution Agents And Target Profiles': '执行主体与目标画像',
        'Ordering And Synchronization': '顺序与同步',
        'Memory Model': '内存模型',
        'Consistency Baseline': '一致性基线',
        'Producer Consumer Ordering': '生产者-消费者顺序',
        'State And Types': '状态与类型',
        'Type System': '类型系统',
        'Location Intent And Legality': '位置意图与合法性',
        'Syntax And Operands': '语法与操作数',
        'Assembly Spelling And Operands': '汇编拼写与操作数',
        'Operands And Attributes': '操作数与属性',
        'Common Conventions': '通用约定',
        'Instruction Overview': '指令总览',
        'Tile Instructions': 'Tile 指令',
        'Vector Instructions': '向量指令',
        'Scalar And Control Instructions': '标量与控制指令',
        'Communication Instructions': '通信指令',
        'System Scheduling Instructions': '系统调度指令',
        'Instruction Set Contracts': '指令集契约',
        'Tile Instruction Set': 'Tile 指令集',
        'Vector Instruction Set': '向量指令集',
        'Scalar And Control Instruction Set': '标量与控制指令集',
        'Communication Instruction Set': '通信指令集',
        'System Scheduling Instruction Set': '系统调度指令集',
        'Tile Instruction Reference': 'Tile 指令参考',
        'Vector ISA Reference': '向量 ISA 参考',
        'Scalar And Control Reference': '标量与控制参考',
        'Communication ISA Reference': '通信 ISA 参考',
        'System Scheduling ISA Reference': '系统调度 ISA 参考',
        'Reference Notes': '参考说明',
        'Glossary': '\u672f\u8bed\u8868',
        'Diagnostics And Illegal Cases': '诊断与非法情形',
        'Portability And Target Profiles': '可移植性与目标画像',
        'Source Of Truth': '事实来源',
        'Instruction Set Overview': '指令集概览',
        'Instruction Set Contract': '指令集契约',
        'Sync And Config': '同步与配置',
        'Elementwise Tile Tile': '逐元素 Tile-Tile',
        'Tile Scalar And Immediate': 'Tile-标量与立即数',
        'Reduce And Expand': '归约与扩展',
        'Memory And Data Movement': '内存与数据搬运',
        'Matrix And Matrix Vector': '矩阵与矩阵-向量',
        'Layout And Rearrangement': '布局与重排',
        'Irregular And Complex': '不规则与复杂指令',
        'Vector Load Store': '向量加载与存储',
        'Predicate And Materialization': '谓词与物化',
        'Unary Vector Instructions': '一元向量指令',
        'Binary Vector Instructions': '二元向量指令',
        'Vector-Scalar Instructions': '向量-标量指令',
        'Conversion Ops': '转换操作',
        'Reduction Instructions': '归约指令',
        'Compare And Select': '比较与选择',
        'Data Rearrangement': '数据重排',
        'SFU And DSA Instructions': 'SFU 与 DSA 指令',
        'Control And Configuration': '控制与配置',
        'Pipeline Sync': '流水线同步',
        'DMA Copy': 'DMA 拷贝',
        'Predicate Load Store': '谓词加载与存储',
        'Predicate Generation And Algebra': '谓词生成与代数',
        'Shared Arithmetic': '共享算术',
        'Shared SCF': '共享 SCF',
        'Communication And Runtime': '通信与运行时',
        'Communication Overview': '通信概览',
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
            var originalTitle = siteTitle.getAttribute('data-original-title') || '';
            if (originalTitle.indexOf('PTO ISA Manual') !== -1) {
                siteTitle.textContent = 'PTO ISA 手册';
            } else if (originalTitle.indexOf('PTO Virtual ISA') !== -1) {
                siteTitle.textContent = 'PTO 虚拟 ISA 手册';
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
