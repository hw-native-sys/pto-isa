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
 * Navigation collapse control
 *
 * This script collapses all second-level navigation items by default,
 * showing only the first-level navigation items.
 * Works with the existing +/- buttons in the ReadTheDocs theme.
 */
(function() {
    'use strict';

    function collapseSecondLevelNav() {
        // 等待 jQuery 加载完成
        if (typeof $ === 'undefined') {
            setTimeout(collapseSecondLevelNav, 100);
            return;
        }

        // 查找所有一级导航项
        var firstLevelItems = $('.wy-menu-vertical li.toctree-l1');

        firstLevelItems.each(function() {
            var $item = $(this);
            var $subMenu = $item.find('> ul');

            if ($subMenu.length === 0) return;

            // 如果当前项不是激活状态，则收起子菜单
            if (!$item.hasClass('current')) {
                $subMenu.hide();
                $item.removeClass('current').attr('aria-expanded', 'false');
            }
        });

        // 更新所有 +/- 按钮的显示
        updateExpandButtons();

        // 使用事件委托监听按钮点击事件
        $('.wy-menu-vertical').off('click.navCollapse').on('click.navCollapse', 'button.toctree-expand', function(e) {
            var $button = $(this);
            var $item = $button.closest('li.toctree-l1');
            var $subMenu = $item.find('> ul');

            if ($subMenu.length === 0) return;

            // 切换子菜单的显示/隐藏
            if ($subMenu.is(':visible')) {
                $subMenu.slideUp(200);
                $item.removeClass('current').attr('aria-expanded', 'false');
                $button.attr('data-expanded', 'false');
            } else {
                $subMenu.slideDown(200);
                $item.addClass('current').attr('aria-expanded', 'true');
                $button.attr('data-expanded', 'true');
            }

            e.stopPropagation();
            e.preventDefault();
        });
    }

    // 更新所有 +/- 按钮的显示
    function updateExpandButtons() {
        if (typeof $ === 'undefined') return;

        $('.wy-menu-vertical li.toctree-l1').each(function() {
            var $item = $(this);
            var $subMenu = $item.find('> ul');
            var $button = $item.find('> a > button.toctree-expand');

            if ($subMenu.length === 0 || $button.length === 0) return;

            // 检查子菜单是否可见
            var isVisible = $subMenu.is(':visible');

            // 根据可见性设置按钮的 data 属性，CSS 可以使用这个属性
            if (isVisible) {
                $button.attr('data-expanded', 'true');
                $item.addClass('current').attr('aria-expanded', 'true');
            } else {
                $button.attr('data-expanded', 'false');
                $item.removeClass('current').attr('aria-expanded', 'false');
            }
        });
    }

    // 初始化
    function init() {
        // 延迟执行，确保主题的 JavaScript 已经加载并初始化完成
        setTimeout(collapseSecondLevelNav, 500);
    }

    // 页面加载完成后初始化
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
