# PTO 虚拟 ISA 架构手册

本站点是 **PTO 虚拟 ISA 架构手册**，采用 PTX/Tile-IR 风格组织：简洁的契约、稳定的术语和明确的架构边界。

## 从这里开始

- [手册入口页面](docs/PTO-Virtual-ISA-Manual_zh.md)
- [手册前言和阅读顺序](manual/index_zh.md)
- [指令参考（每条指令一页）](docs/isa/README_zh.md)
- [编程模型（Tiles/GlobalTensor/Events）](docs/coding/ProgrammingModel_zh.md)
- [抽象机器模型](docs/machine/abstract-machine_zh.md)
- [虚拟 ISA / AS 章节](manual/08-virtual-isa-and-ir_zh.md)
- [字节码 / 工具链章节](manual/09-bytecode-and-toolchain_zh.md)
- [内存排序 / 一致性章节](manual/10-memory-ordering-and-consistency_zh.md)
- [后端配置 / 一致性章节](manual/11-backend-profiles-and-conformance_zh.md)

## 仓库 Markdown 浏览

如果你想浏览 *所有内容*，请使用：

- [完整索引](all-pages_zh.md)



<div class="landing-shell">
  <section class="landing-hero">
    <p class="landing-kicker">Parallel Tile Operation</p>
    <h2>从统一手册阅读 PTO ISA。</h2>
    <p class="landing-lede">
      PTO ISA 是面向多目标后端的虚拟 ISA，覆盖 tile、vector、scalar 与通信相关操作。初次阅读建议先看模型与边界，
      已经熟悉 PTO 的读者可以直接进入中文 ISA 参考树。
    </p>
    <div class="landing-actions">
      <a class="landing-action-card" href="../docs/isa/README_zh.md">
        <span class="landing-action-eyebrow">开始阅读</span>
        <strong>中文 ISA 手册入口</strong>
        <span>从统一目录进入中文手册，再按主题或指令族继续深入。</span>
      </a>
      <a class="landing-action-card" href="../docs/isa/tile/README_zh.md">
        <span class="landing-action-eyebrow">浏览参考</span>
        <strong>中文指令参考树</strong>
        <span>按 tile、vector、scalar、通信等表面浏览具体指令与族契约。</span>
      </a>
      <a class="landing-action-card" href="../index.md">
        <span class="landing-action-eyebrow">切换语言</span>
        <strong>打开 English landing page</strong>
        <span>需要英文原文时，直接切回英文入口；若当前页无中文对照，则语言切换会回到中文或英文入口。</span>
      </a>
    </div>
  </section>

  <section class="landing-grid">
    <article class="landing-panel">
      <h2>先了解 PTO</h2>
      <ul>
        <li><a href="../docs/isa/introduction/what-is-pto-visa_zh.md">什么是 PTO ISA</a></li>
        <li><a href="../docs/isa/introduction/document-structure_zh.md">文档结构</a></li>
        <li><a href="../docs/isa/introduction/goals-of-pto_zh.md">PTO 的目标</a></li>
        <li><a href="../docs/isa/introduction/design-goals-and-boundaries_zh.md">范围与边界</a></li>
        <li><a href="../docs/isa/programming-model/tiles-and-valid-regions_zh.md">编程模型</a></li>
        <li><a href="../docs/isa/machine-model/execution-agents_zh.md">机器模型</a></li>
        <li><a href="../docs/isa/memory-model/consistency-baseline_zh.md">内存模型</a></li>
      </ul>
    </article>

    <article class="landing-panel">
      <h2>按表面浏览</h2>
      <ul>
        <li><a href="../docs/isa/tile/README_zh.md">Tile 指令参考</a></li>
        <li><a href="../docs/isa/vector/README_zh.md">Vector 指令参考</a></li>
        <li><a href="../docs/isa/scalar/README_zh.md">Scalar 与控制参考</a></li>
        <li><a href="../docs/isa/comm/README_zh.md">通信指令集参考</a></li>
        <li><a href="../docs/isa/system/README_zh.md">系统调度指令集参考</a></li>
        <li><a href="../docs/isa/instruction-families/README_zh.md">指令总览</a></li>
      </ul>
    </article>

    <article class="landing-panel">
      <h2>补充说明</h2>
      <ul>
        <li><a href="../docs/isa/syntax-and-operands/assembly-model_zh.md">汇编拼写与操作数</a></li>
        <li><a href="../docs/isa/conventions_zh.md">通用约定</a></li>
        <li><a href="../docs/isa/state-and-types/type-system_zh.md">类型系统</a></li>
        <li><a href="../docs/isa/state-and-types/location-intent-and-legality_zh.md">位置意图与合法性</a></li>
        <li><a href="../docs/isa/reference/format-of-instruction-descriptions_zh.md">指令说明格式</a></li>
        <li><a href="../docs/isa/reference/README_zh.md">参考说明</a></li>
      </ul>
    </article>
  </section>
</div>
