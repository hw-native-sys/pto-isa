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
 * 为中文文档生成 Previous/Next 导航按钮
 * 
 * 策略：完全复制英文页面的导航按钮，只需将链接转换为中文版本
 */
(function() {
    'use strict';

    /**
     * 检测当前页面语言
     */
    function getCurrentLanguage() {
        const path = window.location.pathname;
        if (path.includes('_zh/') || path.endsWith('_zh.html') || path.includes('_zh/index.html')) {
            return 'zh';
        }
        return 'en';
    }

    /**
     * 将中文路径转换为英文路径
     * 
     * 需要处理三种情况：
     * 1. /index_zh/ -> /
     * 2. /manual/index_zh/ -> /manual/
     * 3. /docs/coding/README_zh/ -> /docs/coding/
     * 4. /docs/coding/Tile_zh/ -> /docs/coding/Tile/
     * 5. /docs/isa/index_zh/ -> /docs/isa/ (不是 /docs/isa/index/)
     */
    function zhPathToEnPath(zhPath) {
        // 根目录首页：/index_zh/ -> /
        if (zhPath === '/index_zh/' || zhPath.endsWith('/index_zh/index.html')) {
            return zhPath.replace('/index_zh/', '/');
        }
        
        // Manual 的 index：/manual/index_zh/ -> /manual/
        if (zhPath.includes('/manual/index_zh/')) {
            return zhPath.replace('/manual/index_zh/', '/manual/');
        }
        
        // README 文件：/docs/coding/README_zh/ -> /docs/coding/
        if (zhPath.includes('/README_zh/')) {
            return zhPath.replace('/README_zh/', '/');
        }
        
        // index_zh 文件（除了根目录和 manual）：/docs/isa/index_zh/ -> /docs/isa/
        // 这些是目录的 README.md，编译为 index_zh/
        if (zhPath.includes('/index_zh/')) {
            return zhPath.replace('/index_zh/', '/');
        }
        
        // 普通文件：/docs/coding/Tile_zh/ -> /docs/coding/Tile/
        // /manual/01-overview_zh/ -> /manual/01-overview/
        return zhPath.replace(/_zh\//g, '/');
    }

    /**
     * 将英文链接转换为中文链接
     * 
     * README.md 编译为 README_zh/，普通文件编译为 filename_zh/
     * 必须与 nav-translator.js 中的逻辑保持一致
     * 
     * 重要：需要将相对路径转换为绝对路径，避免基于中文页面路径解析导致错误
     */
    function enLinkToZhLink(enLink, currentEnPath) {
        // 首先将英文相对链接转换为绝对路径
        let absoluteEnLink = enLink;
        if (!enLink.startsWith('/') && !enLink.startsWith('http')) {
            // 相对路径，需要基于英文页面路径解析
            // currentEnPath 例如：/manual/01-overview/
            // enLink 例如：../02-machine-model/
            
            // 创建一个临时 URL 对象来解析相对路径
            const baseUrl = 'http://dummy.com' + currentEnPath;
            try {
                const resolvedUrl = new URL(enLink, baseUrl);
                absoluteEnLink = resolvedUrl.pathname;
            } catch (e) {
                console.error('Failed to resolve relative link:', enLink, 'from', currentEnPath);
                absoluteEnLink = enLink;
            }
        }
        
        // 现在 absoluteEnLink 是绝对路径，例如：/manual/02-machine-model/
        let newHref = absoluteEnLink;
        
        // 特殊处理：根目录首页
        if (newHref === '/' || newHref === '/index.html') {
            return '/index_zh/';
        }
        // manual 目录的 index：/manual/ -> /manual/index_zh/
        else if (newHref === '/manual/' || newHref.endsWith('/manual/')) {
            return newHref.replace(/\/manual\/$/, '/manual/index_zh/');
        }
        // 其他以 / 结尾的路径
        else if (newHref.endsWith('/')) {
            const pathParts = newHref.split('/').filter(p => p);
            const lastPart = pathParts[pathParts.length - 1];
            
            // 判断是否是 README（与 nav-translator.js 保持一致）
            const isReadme = (
                // 顶级目录
                ['coding', 'isa', 'ir', 'grammar', 'machine', 'kernels', 'tests', 'docs', 'scripts', 'demos', 'include', 'cmake'].includes(lastPart) ||
                // tutorials 子目录
                lastPart === 'tutorials' ||
                // 其他已知的 README 目录
                lastPart === 'script' ||
                lastPart === 'package' ||
                lastPart === 'custom' ||
                lastPart === 'baseline' ||
                lastPart === 'add' ||
                lastPart === 'gemm_basic' ||
                lastPart === 'flash_atten' ||
                lastPart === 'gemm_performance' ||
                lastPart === 'a2a3' ||
                lastPart === 'a5' ||
                lastPart === 'kirin9030' ||
                lastPart === 'npu' ||
                lastPart === 'pto' ||
                lastPart === 'reference'
            );
            
            if (isReadme) {
                // README: /docs/coding/ -> /docs/coding/README_zh/
                return newHref.replace(/\/$/, '/README_zh/');
            } else {
                // 普通文件: /docs/coding/Tile/ -> /docs/coding/Tile_zh/
                // /manual/01-overview/ -> /manual/01-overview_zh/
                return newHref.replace(/\/([^\/]+)\/$/, '/$1_zh/');
            }
        }
        
        return newHref;
    }

    /**
     * 从对应的英文页面获取导航按钮
     */
    function getEnglishNavigation() {
        const zhPath = window.location.pathname;
        const enPath = zhPathToEnPath(zhPath);
        
        console.log('Chinese path:', zhPath);
        console.log('English path:', enPath);
        
        return fetch(enPath)
            .then(response => {
                if (!response.ok) {
                    console.error('English page not found:', enPath, response.status);
                    throw new Error('English page not found');
                }
                return response.text();
            })
            .then(html => {
                const parser = new DOMParser();
                const doc = parser.parseFromString(html, 'text/html');
                
                // 查找导航按钮
                const navButtons = doc.querySelector('.rst-footer-buttons');
                if (!navButtons) {
                    console.log('No navigation buttons found in English page');
                    return null;
                }
                
                const prevLink = navButtons.querySelector('a.float-left');
                const nextLink = navButtons.querySelector('a.float-right');
                
                console.log('Found navigation:', {
                    prev: prevLink ? prevLink.getAttribute('href') : null,
                    next: nextLink ? nextLink.getAttribute('href') : null
                });
                
                return {
                    prev: prevLink ? {
                        href: prevLink.getAttribute('href'),
                        title: prevLink.getAttribute('title') || prevLink.textContent.trim()
                    } : null,
                    next: nextLink ? {
                        href: nextLink.getAttribute('href'),
                        title: nextLink.getAttribute('title') || nextLink.textContent.trim()
                    } : null
                };
            })
            .catch(error => {
                console.error('Failed to fetch English navigation:', error);
                return null;
            });
    }

    /**
     * 生成导航按钮
     */
    function generateNavigationButtons() {
        const currentLang = getCurrentLanguage();
        
        // 只为中文页面生成
        if (currentLang !== 'zh') {
            return;
        }
        
        // 检查是否已经有导航按钮
        const existingNav = document.querySelector('.rst-footer-buttons');
        if (existingNav && existingNav.querySelector('a')) {
            return;
        }
        
        // 获取当前中文页面路径，并转换为英文路径
        const zhPath = window.location.pathname;
        const enPath = zhPathToEnPath(zhPath);
        
        // 从英文页面获取导航信息
        getEnglishNavigation().then(navInfo => {
            if (!navInfo) {
                return;
            }
            
            // 转换链接为中文版本（传入英文页面路径用于解析相对路径）
            const prevPage = navInfo.prev ? {
                href: enLinkToZhLink(navInfo.prev.href, enPath),
                title: navInfo.prev.title
            } : null;
            
            const nextPage = navInfo.next ? {
                href: enLinkToZhLink(navInfo.next.href, enPath),
                title: navInfo.next.title
            } : null;
            
            // 创建导航按钮
            createNavigationButtons(prevPage, nextPage);
        });
    }

    /**
     * 创建导航按钮 DOM
     */
    function createNavigationButtons(prevPage, nextPage) {
        // 查找或创建 footer
        let footer = document.querySelector('footer');
        if (!footer) {
            const mainContent = document.querySelector('.wy-nav-content');
            if (!mainContent) return;
            
            footer = document.createElement('footer');
            mainContent.appendChild(footer);
        }
        
        // 创建导航按钮容器
        const navButtons = document.createElement('div');
        navButtons.className = 'rst-footer-buttons';
        navButtons.setAttribute('role', 'navigation');
        navButtons.setAttribute('aria-label', 'Footer Navigation');
        
        // Previous 按钮
        if (prevPage) {
            const prevBtn = document.createElement('a');
            prevBtn.href = prevPage.href;
            prevBtn.className = 'btn btn-neutral float-left';
            prevBtn.title = prevPage.title;
            prevBtn.innerHTML = '<span class="icon icon-circle-arrow-left"></span> 上一页';
            navButtons.appendChild(prevBtn);
        }
        
        // Next 按钮
        if (nextPage) {
            const nextBtn = document.createElement('a');
            nextBtn.href = nextPage.href;
            nextBtn.className = 'btn btn-neutral float-right';
            nextBtn.title = nextPage.title;
            nextBtn.innerHTML = '下一页 <span class="icon icon-circle-arrow-right"></span>';
            navButtons.appendChild(nextBtn);
        }
        
        // 插入到 footer 开头
        footer.insertBefore(navButtons, footer.firstChild);
        
        // 同时更新底部的 rst-versions 区域（如果存在）
        updateBottomNavigation(prevPage, nextPage);
    }

    /**
     * 更新底部导航区域
     */
    function updateBottomNavigation(prevPage, nextPage) {
        const rstVersions = document.querySelector('.rst-versions .rst-current-version');
        if (!rstVersions) return;
        
        // 清空现有内容
        rstVersions.innerHTML = '';
        
        // 添加 Previous 链接
        if (prevPage) {
            const prevSpan = document.createElement('span');
            const prevLink = document.createElement('a');
            prevLink.href = prevPage.href;
            prevLink.style.color = '#fcfcfc';
            prevLink.textContent = '« 上一页';
            prevSpan.appendChild(prevLink);
            rstVersions.appendChild(prevSpan);
        }
        
        // 添加 Next 链接
        if (nextPage) {
            const nextSpan = document.createElement('span');
            nextSpan.style.float = 'right';
            const nextLink = document.createElement('a');
            nextLink.href = nextPage.href;
            nextLink.style.color = '#fcfcfc';
            nextLink.textContent = '下一页 »';
            nextSpan.appendChild(nextLink);
            rstVersions.appendChild(nextSpan);
        }
    }

    /**
     * 初始化
     */
    function init() {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', generateNavigationButtons);
        } else {
            generateNavigationButtons();
        }
    }

    // 启动
    init();
})();
