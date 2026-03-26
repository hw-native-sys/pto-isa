This file lists some rules and limitations on the implementation of this library for pto-isa developers.
这篇文档列出一些针对pto-isa库开发者在auto模式下的编程规则和限制。

不遵守这些规则可能导致以下任何后果：

1. 无法编译（包括源码层面编译报错或者编译器挂掉）
2. 结果错误（例如，精度问题）
3. 差劲的性能

# 1 - 记住：`pto::(Conv)Tile::data()` 在auto模式下返回的是vector类型而不是指针

`Tile` struct的`.data()`成员函数返回类型为`TileDType`，其在manual和auto模式下定义是不同的：manual模式下就是指针类型，然而在auto模式下是clang的vector类型。
请务必将此牢记于心，避免在调用tile function以外的地方直接调用`.data()`来尝试获得Tile的地址指针。

# 2 - 避免在struct/class的成员上使用default-initializer

在C++中，直接在struct/class成员上默认初始化是很常见的：

```cpp
struct ConvTile {
public:
    ...

    int shape[ConvTileDetail::MAX_CONVTILE_DIM] = {1};
};
```

然而这样会对AUTO编译器产生问题：编译器内的SROA pass无法消除此成员对应的`AllocaInst`指令（以及其对应的`load`和`store`指令），会对后续的内存分配产生不利影响。请尽量避免默认初始化struct/class成员（至少，auto模式下不要）:

```cpp
#ifdef __PTO_AUTO__
    // In auto mode, do not have default initialization in the class definition itself for its members
    int shape[ConvTileDetail::MAX_CONVTILE_DIM];
#else
    int shape[ConvTileDetail::MAX_CONVTILE_DIM] = {1};
#endif
```

虽然是用C++，但是我们还是鼓励鼓励尽量使用POD（plain old data）来靠近C语言。

# 3 - 在tile function以及被其调用的函数内，依然需要开发者手动同步

TL;DR:

- 在tile function内（包括被其调用的函数内）调用`set_flag`，`wait_flag`或者`pipe_barrier`
- 在其他地方尽量调用`PtoSetWaitFlag`或者`TSYNC`

原因：PTO编译器的分析以及优化都是建立在tile这一抽象层级上的，而tile function是这一抽象层级的最后一层；一旦进入tile function，就脱离了tile的层级范围，进入了CCE的领域。因此，tile function内部对于PTO编译器来说就是一个黑盒子，完全不关心里面是什么。

因此，如果在tile function内部需要同步，依然需要库开发者手动插入同步，因为进入这一层已经超出了PTO编译器的使用范围。所以在tile function内部不能使用`PtoSetWaitFlag`或者`TSYNC`，因为它们在auto模式下都是no-op。

# 4 - 避免使用`TASSIGN`来实现PTO指令

目前，有一些PTO指令的内部实现直接调用了`TASSIGN_IMPL`。这在auto模式下是行不通的。

如果开发者的意图是想alias两个tile，那应该使用`TRESHAPE`或者`TSUBVIEW`来实现；其余任何用途在auto模式下都不能达到正确的结果。

比如，如果开发者直接调用`TASSIGN`来分配tile的地址，而这个分配的地址是基于一些算法或者代码逻辑实现的，那auto编译器肯定无法感知到具体的算法逻辑而实现和manual模式下相同的内存分配结果。

毕竟，auto模式下的自动内存分配是完全基于每个tile的liveness analysis的，没有其他任何context信息。这就是为什么目前TPUSH和TPOP的实现在auto模式下不可能正确运行。

# 5 - 一些`*_IMPL`层函数的规则

Some consistency must be ensured for `*_IMPL` and tile function interface:
`*_IMPL`层接口需要遵守如下规则：

- 函数签名必须有`PTO_INTERNAL`接口
- 其内部实现必须直接调用tile function；如果需要调用其他非tile function的函数则必须被inline。
- 确保对每个Tile调用`.data()`的同时调用tile function，或者确保`.data()`返回的值是reference。示例：

```cpp
TExp(dstTile.data(), srcTile.data()); // correct

auto dst = dstTile.data(); // wrong: return by value
auto &src = srcTile.data(); // correct: return by reference
TExp(dst, src);
```

- The function signature must have `PTO_INTERNAL` macro
- Its implementation should directly call tile functions inside, don't call any non-tile functions unless they're inlined.
- Always call `.data()` function to pass into tile functions

# 6 - tile function层接口的规则

- 确保函数参数中，使用`typename <...>::TileDType`而不是`typename <...>::DType *`来表示tile类型
- 确保这些`typename <...>::TileDType`参数是pass-by-value, 而不是pass-by-reference或者pass-by-pointer的
- 确保所有`typename <...>::TileDType`参数都有`__in__`或者`__out__`标记
- 在实现中，确保对每个`typename <...>::TileDType`参数都调用`__cce_get_tile_ptr`来获取其buffer指针，除非是unused的参数
- 返回类型必须为void，不然会违反编译器对于TF接口的假设而且是未定义行为。如果开发者想让tile function有返回值，请使用pass-by-reference的入参，即使对一个scalar输出也是如此。

# 7 - 在TF层以上尽量避免动态控制流

动态控制流对自动同步一直是一个巨大的挑战。我们鼓励开发者要么在tile function层以上避免使用动态控制流（如果不是必须），或者将控制流移到tile function内。

tile function层以前有动态控制流的典型接口包括`TROWEXPANDDIV_IMPL`和`TMULS_IMPL`。
