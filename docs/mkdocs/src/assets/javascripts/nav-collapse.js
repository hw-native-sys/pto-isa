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
 * Navigation collapse control.
 *
 * The ReadTheDocs theme already renders expandable first-level sections. We
 * keep that behavior, but remove the previous jQuery + timeout dependency and
 * manage the expanded state directly from the DOM.
 */
(function () {
    'use strict';

    function directChild(element, selector) {
        return Array.from(element.children).find(function (child) {
            return child.matches(selector);
        }) || null;
    }

    function hydrateSectionLinks() {
        document.querySelectorAll('.wy-menu-vertical li').forEach(function (item) {
            var anchor = directChild(item, 'a.reference.internal:not([href])');
            var subMenu = directChild(item, 'ul');

            if (!anchor || !subMenu) {
                return;
            }

            var firstChildLink = Array.from(subMenu.querySelectorAll('a.reference.internal[href]')).find(function (link) {
                var href = link.getAttribute('href');
                return href && !href.startsWith('#');
            });

            if (!firstChildLink) {
                return;
            }

            anchor.setAttribute('href', firstChildLink.getAttribute('href'));
            anchor.setAttribute('data-nav-section-link', 'true');
        });
    }

    function setExpanded(item, expanded) {
        var subMenu = directChild(item, 'ul');
        if (!subMenu) {
            return;
        }

        var button = item.querySelector(':scope > a > button.toctree-expand');
        var initialCurrent = item.getAttribute('data-initial-current') === 'true';

        subMenu.style.display = expanded ? 'block' : 'none';
        item.setAttribute('aria-expanded', expanded ? 'true' : 'false');

        if (button) {
            button.setAttribute('data-expanded', expanded ? 'true' : 'false');
            button.setAttribute('aria-expanded', expanded ? 'true' : 'false');
        }

        if (!initialCurrent) {
            item.classList.toggle('current', expanded);
        }
    }

    function initializeTree() {
        document.querySelectorAll('.wy-menu-vertical li.toctree-l1').forEach(function (item) {
            var subMenu = directChild(item, 'ul');
            if (!subMenu) {
                return;
            }

            var isCurrent = item.classList.contains('current') || item.classList.contains('on');
            item.setAttribute('data-initial-current', isCurrent ? 'true' : 'false');
            setExpanded(item, isCurrent);
        });
    }

    function bindButtons() {
        document.querySelectorAll('.wy-menu-vertical li.toctree-l1 > a > button.toctree-expand').forEach(function (button) {
            button.addEventListener('click', function (event) {
                var item = button.closest('li.toctree-l1');
                if (!item) {
                    return;
                }

                var expanded = button.getAttribute('data-expanded') === 'true';
                setExpanded(item, !expanded);
                event.preventDefault();
                event.stopPropagation();
            });
        });
    }

    function init() {
        hydrateSectionLinks();
        initializeTree();
        bindButtons();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
