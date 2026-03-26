# auto模式总览

## 范围

这篇文档提供一个auto模式的概览。

## auto模式是什么

AUTO模式是一个新的PTO的编译模式。编译器负责自动为Tile分配内存和插入同步指令。和manual模式下编程大致相同，只是程序员不需要手动调用`TASSIGN`来分配内存，以及手动调用同步指令来同步pipe。

## 为什么使用auto mode？

auto模式的目标提高用户的编程效率，同时保持相对较好的性能（与一个manual模式下手动优化的代码相比）。

主要功能包括：

* 自动在不同硬件pipe之间插入同步指令
* 对Tile对象自动分配内存

## AUTO模式特性

接下来的部分会介绍auto模式的特性，作为程序员可用的另一种编程模型。

重要:
一般来说，每一个PTO指令的实现，都应该拥有以下不同层级的API（从最高层到最低）：

* 用户层（kernel开发者可调用的公有的，最高层级API）
* IMPL层
* TF （Tile Function）层
* CCE实现层（例如，vector function，SMIT，等等）

PTO编译器，是在Tile这一抽象层级工作的。这意味着，**以下列出的所有特性都只在TF层以上运作**，因为TF层接口是Tile抽象层级的最后一层；一旦进入了tile function，就脱离了Tile抽象层级，而进入了裸指针和CCE intrinsics的层级（CCE的领域）。因此，对PTO编译器来说，tile function是一个完全的黑盒子，PTO编译器的功能不会在tile function运作（那是CCE编译器处理的部分了）。

**更具体来说，所有以下特性，只能在tile function层以上运作，而不会进入tile function内。**

### Tile的自动liveness分析

在auto模式下，编译器会分析每一个Tile的liveness。这个liveness分析是auto模式的核心，是给以下功能实现提供支持。

### 自动同步

在manual模式下，程序员需要熟悉昇腾硬件不同pipe之间异步运行的特性，并运用PTO的Event编程模型，在正确的地点手动插入同步指令，来保证正确的结果和高性能。这非常繁琐，且很容易出错。

auto模式编译器给程序员省去了这个麻烦。编译器会代替程序员自动在正确的位置插入正确的同步指令，确保正确的结果以及相对较好的性能。

### Tile内存分配

在manual模式下，程序员需要手动调用`TASSIGN`来为每一个Tile对象分配对应硬件buffer上的内存地址。

然而在auto模式下这也不需要了。对于每一个定义的Tile对象，编译器会自动替程序员在正确的buffer上分配地址。

## 使用Ascend CANN编译auto模式代码

要用auto模式编译你的kernel，你只需要加上一条编译命令：`--cce-enable-pto-passes`，来使能auto模式编译。

### 示例（device侧编译）

要想编译一个device侧的kernel，请确保：

* 针对你的Soc使用正确的`--cce-aicore-arch...`（比如，`dav-c220-vec`，`dav-c310-vec`，等等）。

示例：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash

bisheng -c -x cce -O2 --cce-aicore-only \
  --cce-aicore-arch=dav-c310-vec \
  -std=c++17 \
  --cce-enable-pto-passes \
  kernel.cpp -o kernel.o
```
