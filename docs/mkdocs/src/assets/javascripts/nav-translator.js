// --------------------------------------------------------------------------------
// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// --------------------------------------------------------------------------------

// 导航栏中英文映射表
const NAV_TRANSLATIONS = {
    // 顶级导航
    'Home': '首页',
    'PTO Virtual ISA Manual': 'PTO 虚拟 ISA 手册',
    'Programming Model': '编程模型',
    'ISA Reference': 'ISA 参考',
    'Machine Model': '机器模型',
    'Examples': '示例',
    'Documentation': '文档',
    'Full Index': '完整索引',
    
    // PTO Virtual ISA Manual 子项
    'Preface': '前言',
    'Overview': '概述',
    'Execution Model': '执行模型',
    'State and Types': '状态与类型',
    'Tiles and GlobalTensor': 'Tile 与 GlobalTensor',
    'Synchronization': '同步',
    'PTO Assembly (PTO-AS)': 'PTO 汇编 (PTO-AS)',
    'Instruction Set (overview)': '指令集（概述）',
    'Programming Guide': '编程指南',
    'Virtual ISA and IR': '虚拟 ISA 与 IR',
    'Bytecode and Toolchain': '字节码与工具链',
    'Memory Ordering and Consistency': '内存顺序与一致性',
    'Backend Profiles and Conformance': '后端配置与一致性',
    'Glossary': '术语表',
    'Instruction Contract Template': '指令契约模板',
    'Diagnostics Taxonomy': '诊断分类',
    'Instruction Family Matrix': '指令族矩阵',
    
    // Programming Model 子项
    'Tile': 'Tile',
    'GlobalTensor': 'GlobalTensor',
    'Scalar': 'Scalar',
    'Event': 'Event',
    'Tutorial': '教程',
    'Tutorials': '教程集',
    'Example: Vec Add': '示例：向量加法',
    'Example: Row Softmax': '示例：行 Softmax',
    'Example: GEMM': '示例：GEMM',
    'Optimization': '优化',
    'Debugging': '调试',
    
    // ISA Reference 子项
    'ISA index': 'ISA 索引',
    'PTO IR ops index': 'PTO IR 操作索引',
    'ISA conventions': 'ISA 约定',
    'PTO ISA table': 'PTO ISA 表',
    'Intrinsics header': '内建函数头文件',
    'PTO-AS spec': 'PTO-AS 规范',
    'Grammar conventions': '语法约定',
    'Grammar index': '语法索引',
    
    // Machine Model 子项
    'Abstract machine': '抽象机器',
    'Machine index': '机器索引',
    
    // Examples 子项
    'Kernels index': '算子索引',
    'GEMM performance kernel': 'GEMM 性能算子',
    'Flash Attention kernel': 'Flash Attention 算子',
    'Baseline add demo': '基础加法示例',
    'Baseline GEMM demo': '基础 GEMM 示例',
    'Tests index': '测试索引',
    'Test scripts': '测试脚本',
    
    // Documentation 子项
    'Docs index': '文档索引',
    'Build this site': '构建本站点',
    'Root README': '根目录 README'
};

// 导航栏翻译函数（内部实现）
function translateNavigationInternal(targetLang) {
    if (targetLang !== 'zh') {
        return; // 只翻译为中文
    }
    
    console.log('=== translateNavigation called ===');
    
    // 查找所有导航链接
    const navLinks = document.querySelectorAll('.wy-menu-vertical a, nav a, .toctree-l1 > a, .toctree-l2 > a');
    
    navLinks.forEach(link => {
        const originalText = link.textContent.trim();
        
        // 跳过空文本
        if (!originalText) return;
        
        // 保存原始英文文本（如果还没保存）
        if (!link.hasAttribute('data-original-text')) {
            link.setAttribute('data-original-text', originalText);
        }
        
        // 翻译文本
        const origText = link.getAttribute('data-original-text');
        if (NAV_TRANSLATIONS[origText]) {
            link.textContent = NAV_TRANSLATIONS[origText];
        }
        
        // 修改链接指向中文版本
        // 获取当前 href
        const currentHref = link.getAttribute('href');
        
        // 总是重新计算并保存原始英文 href
        // 这样可以处理 MkDocs 动态修改链接的情况
        if (currentHref && (currentHref.includes('_zh/') || currentHref.includes('_zh.html'))) {
            // 当前是中文链接，转换回英文
            let enHref = currentHref;
            
            if (enHref.includes('/index_zh/')) {
                enHref = enHref.replace('/index_zh/', '/');
            } else if (enHref.includes('manual/index_zh/')) {
                enHref = enHref.replace(/manual\/index_zh\//, 'manual/');
            } else if (enHref.includes('/README_zh/')) {
                enHref = enHref.replace('/README_zh/', '/');
            } else if (enHref.includes('README_zh/')) {
                // 处理相对路径：./README_zh/ -> ./
                enHref = enHref.replace(/README_zh\//, '');
            } else if (enHref.includes('index_zh/')) {
                // 处理相对路径：./index_zh/ -> ./
                enHref = enHref.replace(/index_zh\//, '');
            } else {
                enHref = enHref.replace(/_zh\//g, '/');
            }
            
            console.log('Converting Chinese link to English:', currentHref, '->', enHref);
            link.setAttribute('data-original-href', enHref);
        } else if (currentHref && !link.hasAttribute('data-original-href')) {
            // 当前是英文链接，且还没保存，直接保存
            console.log('Saving English link:', currentHref);
            link.setAttribute('data-original-href', currentHref);
        }
        
        // 使用原始 href 进行转换
        const href = link.getAttribute('data-original-href');
        
        // 如果没有保存原始 href，跳过转换
        if (!href) {
            console.warn('No original href found for link:', originalText, 'current href:', currentHref);
            return;
        }
        
        console.log('Translating link:', originalText, 'from', href);
        
        if (href && !href.startsWith('#') && !href.startsWith('http')) {
            let newHref = href;
            
            // 特殊处理：根目录首页
            if (newHref === '../..' || newHref === '../../' || 
                newHref === '../../..' || newHref === '../../../' ||
                newHref === '/' || newHref === '/index.html' ||
                newHref === '/.' || newHref === '/..' || newHref === '/../') {
                newHref = '/index_zh/';
            }
            // 当前目录（README）：./ -> ./README_zh/
            else if (newHref === './' || newHref === '.') {
                newHref = './README_zh/';
            }
            // 父目录：../ 的处理需要根据上下文判断
            // 如果链接文本和当前页面标题相同，说明是指向当前目录的自引用
            // 例如在 /docs/isa/README_zh/ 页面，"ISA index" 链接指向 ../（即 /docs/isa/）
            else if (newHref === '../' || newHref === '..') {
                // 获取当前页面路径
                const currentPath = window.location.pathname;
                
                // 如果当前在 README_zh 或 index_zh 页面，且链接指向父目录
                // 检查是否是自引用（链接文本包含当前页面的关键词）
                if ((currentPath.includes('/README_zh/') || currentPath.includes('/index_zh/')) && 
                    origText && (
                        currentPath.toLowerCase().includes(origText.toLowerCase().replace(/\s+/g, '')) ||
                        origText.toLowerCase().includes('index') ||
                        origText.toLowerCase().includes('索引')
                    )) {
                    // 这是自引用，使用当前页面的绝对路径
                    newHref = currentPath;
                } else {
                    // 真正的父目录引用
                    newHref = '../index_zh/';
                }
            }
            // manual 目录的 index：../../manual/ -> ../../manual/index_zh/
            else if (newHref.endsWith('/manual/') || newHref === 'manual/') {
                newHref = newHref.replace(/manual\/$/, 'manual/index_zh/');
            }
            // 其他以 / 结尾的路径
            else if (newHref.endsWith('/')) {
                // 根据导航配置，判断是 README 还是普通文件
                // README 的特征：
                // - docs/coding/ (Overview)
                // - docs/isa/ (ISA index)
                // - docs/ir/ (PTO IR ops index)
                // - docs/grammar/ (Grammar index)
                // - docs/machine/ (Machine index)
                // - kernels/ (Kernels index)
                // - tests/ (Tests index)
                // - docs/coding/tutorials/ (Tutorials)
                
                // 普通文件的特征：
                // - 包含连字符的名称（如 01-overview, vec-add）
                // - 特定的文件名（Tile, GlobalTensor, Scalar, Event, ProgrammingModel, tutorial, opt, debug）
                
                const pathParts = newHref.split('/').filter(p => p && p !== '..');
                const lastPart = pathParts[pathParts.length - 1];
                
                // 判断是否是 README
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
                    // README: docs/coding/ -> docs/coding/README_zh/
                    newHref = newHref.replace(/\/$/, '/README_zh/');
                } else {
                    // 普通文件: Tile/ -> Tile_zh/
                    newHref = newHref.replace(/\/([^\/]+)\/$/, '/$1_zh/');
                }
            }
            
            console.log('Setting href:', originalText, 'from', href, 'to', newHref);
            link.setAttribute('href', newHref);
        }
    });
    
    // 翻译导航栏的大标题（caption-text）
    const captionTexts = document.querySelectorAll('.caption-text');
    captionTexts.forEach(caption => {
        const originalText = caption.textContent.trim();
        
        if (!originalText) return;
        
        // 保存原始文本
        if (!caption.hasAttribute('data-original-text')) {
            caption.setAttribute('data-original-text', originalText);
        }
        
        const origText = caption.getAttribute('data-original-text');
        if (NAV_TRANSLATIONS[origText]) {
            // 使用 textContent 而不是 innerHTML，避免破坏 DOM 结构
            caption.textContent = NAV_TRANSLATIONS[origText];
        }
        
        // 阻止标题的点击事件（标题不应该是可点击的）
        const parent = caption.closest('p.caption');
        if (parent && !parent.hasAttribute('data-click-protected')) {
            parent.setAttribute('data-click-protected', 'true');
            parent.addEventListener('click', function(e) {
                // 如果点击的是标题本身（不是子链接），阻止默认行为
                if (e.target === caption || e.target === parent || e.target.classList.contains('caption-text')) {
                    e.preventDefault();
                    e.stopPropagation();
                    console.log('Navigation title click prevented');
                }
            }, true); // 使用捕获阶段
        }
    });
    
    // 翻译站点标题
    const siteTitle = document.querySelector('.wy-side-nav-search a, .navbar-brand');
    if (siteTitle) {
        // 保存原始标题
        if (!siteTitle.hasAttribute('data-original-title')) {
            siteTitle.setAttribute('data-original-title', siteTitle.textContent);
        }
        
        const origTitle = siteTitle.getAttribute('data-original-title');
        if (origTitle && origTitle.includes('PTO Virtual ISA')) {
            siteTitle.textContent = 'PTO 虚拟 ISA 架构手册';
        }
    }
}

// 恢复英文导航（内部实现）
function restoreEnglishNavigationInternal() {
    // 查找所有导航链接
    const navLinks = document.querySelectorAll('.wy-menu-vertical a, nav a, .toctree-l1 > a, .toctree-l2 > a');
    
    navLinks.forEach(link => {
        // 恢复原始英文文本
        const originalText = link.getAttribute('data-original-text');
        if (originalText) {
            link.textContent = originalText;
        }
        
        // 恢复原始英文链接
        const originalHref = link.getAttribute('data-original-href');
        if (originalHref) {
            link.setAttribute('href', originalHref);
        }
    });
    
    // 恢复导航栏的大标题
    const captionTexts = document.querySelectorAll('.caption-text');
    captionTexts.forEach(caption => {
        const originalText = caption.getAttribute('data-original-text');
        if (originalText) {
            // 使用 textContent 恢复，保持 DOM 结构
            caption.textContent = originalText;
        }
    });
    
    // 恢复站点标题
    const siteTitle = document.querySelector('.wy-side-nav-search a, .navbar-brand');
    if (siteTitle) {
        const originalTitle = siteTitle.getAttribute('data-original-title');
        if (originalTitle) {
            siteTitle.textContent = originalTitle;
        }
    }
}

// 监控导航链接的变化，防止被其他代码修改
let linkObservers = []; // 保存所有 observers，以便清理

function protectNavigationLinks() {
    // 清理之前的 observers
    linkObservers.forEach(observer => observer.disconnect());
    linkObservers = [];
    
    const navLinks = document.querySelectorAll('.wy-menu-vertical a[data-original-href]');
    
    console.log('Protecting', navLinks.length, 'navigation links');
    
    navLinks.forEach(link => {
        // 方法 1：使用 MutationObserver 监控 href 属性的变化
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.type === 'attributes' && mutation.attributeName === 'href') {
                    const currentHref = link.getAttribute('href');
                    const originalHref = link.getAttribute('data-original-href');
                    
                    // 如果当前语言是中文，计算正确的中文链接
                    const preferredLang = localStorage.getItem('preferred_language') || 'en';
                    if (preferredLang === 'zh' && originalHref) {
                        let expectedHref = calculateChineseHref(originalHref, link);
                        
                        // 如果 href 被修改为英文链接，立即恢复为中文链接
                        if (currentHref !== expectedHref && !currentHref.includes('_zh')) {
                            console.log('Link modified! Restoring:', currentHref, '->', expectedHref);
                            link.setAttribute('href', expectedHref);
                        }
                    }
                }
            });
        });
        
        // 开始监控
        observer.observe(link, {
            attributes: true,
            attributeFilter: ['href']
        });
        
        linkObservers.push(observer);
        
        // 方法 2：在点击时拦截并确保链接正确
        link.addEventListener('click', function(e) {
            const preferredLang = localStorage.getItem('preferred_language') || 'en';
            if (preferredLang === 'zh') {
                const currentHref = this.getAttribute('href');
                const originalHref = this.getAttribute('data-original-href');
                
                if (originalHref) {
                    const expectedHref = calculateChineseHref(originalHref, this);
                    
                    // 如果当前 href 不是中文链接，阻止默认行为并手动跳转
                    if (currentHref !== expectedHref) {
                        console.log('Click intercepted! Redirecting from', currentHref, 'to', expectedHref);
                        e.preventDefault();
                        e.stopPropagation();
                        window.location.href = expectedHref;
                    }
                }
            }
        }, true); // 使用捕获阶段，优先于其他事件处理器
    });
}

// 计算中文链接的辅助函数
function calculateChineseHref(originalHref, linkElement) {
    let expectedHref = originalHref;
    
    // 转换为中文链接
    if (expectedHref === '../..' || expectedHref === '../../' || 
        expectedHref === '../../..' || expectedHref === '../../../' ||
        expectedHref === '/' || expectedHref === '/index.html' ||
        expectedHref === '/.' || expectedHref === '/..' || expectedHref === '/../' ||
        expectedHref === '..' || expectedHref === '..') {
        expectedHref = '/index_zh/';
    } else if (expectedHref === './' || expectedHref === '.') {
        expectedHref = './README_zh/';
    } else if (expectedHref === '../' || expectedHref === '..') {
        // 父目录：需要根据上下文判断
        const currentPath = window.location.pathname;
        const origText = linkElement ? linkElement.getAttribute('data-original-text') : '';
        
        if ((currentPath.includes('/README_zh/') || currentPath.includes('/index_zh/')) && 
            origText && (
                currentPath.toLowerCase().includes(origText.toLowerCase().replace(/\s+/g, '')) ||
                origText.toLowerCase().includes('index') ||
                origText.toLowerCase().includes('索引')
            )) {
            // 自引用，使用当前页面的绝对路径
            expectedHref = currentPath;
        } else {
            // 真正的父目录引用
            expectedHref = '../index_zh/';
        }
    } else if (expectedHref.endsWith('/manual/') || expectedHref === 'manual/') {
        expectedHref = expectedHref.replace(/manual\/$/, 'manual/index_zh/');
    } else if (expectedHref.endsWith('/')) {
        const pathParts = expectedHref.split('/').filter(p => p && p !== '..');
        const lastPart = pathParts[pathParts.length - 1];
        
        const isReadme = (
            ['coding', 'isa', 'ir', 'grammar', 'machine', 'kernels', 'tests', 'docs', 'scripts', 'demos', 'include', 'cmake'].includes(lastPart) ||
            lastPart === 'tutorials' ||
            lastPart === 'script' || lastPart === 'package' || lastPart === 'custom' ||
            lastPart === 'baseline' || lastPart === 'add' || lastPart === 'gemm_basic' ||
            lastPart === 'flash_atten' || lastPart === 'gemm_performance' ||
            lastPart === 'a2a3' || lastPart === 'a5' || lastPart === 'kirin9030' ||
            lastPart === 'npu' || lastPart === 'pto' || lastPart === 'reference'
        );
        
        if (isReadme) {
            expectedHref = expectedHref.replace(/\/$/, '/README_zh/');
        } else {
            expectedHref = expectedHref.replace(/\/([^\/]+)\/$/, '/$1_zh/');
        }
    }
    
    return expectedHref;
}

// 停止保护（切换回英文时）
function stopProtectingLinks() {
    linkObservers.forEach(observer => observer.disconnect());
    linkObservers = [];
}

// 导出函数供语言切换器使用
window.translateNavigation = function(targetLang) {
    translateNavigationInternal(targetLang);
    if (targetLang === 'zh') {
        // 延迟一点启动保护，确保翻译完成
        setTimeout(protectNavigationLinks, 100);
    }
};

window.restoreEnglishNavigation = function() {
    stopProtectingLinks();
    restoreEnglishNavigationInternal();
};

