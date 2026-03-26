This file lists some rules and limitations on the implementation of this library for pto-isa developers.

Not following the rules can result in any of the following:

1. Can't compile (including source-code level compile errors and crash in compiler)
2. Functionally incorrect (e.g., precision issues)
3. Bad performance

# 1 - Remember that `pto::(Conv)Tile::data()` returns vector type instead of pointer type in auto mode

The return type of `.data()` member function is `TileDType`, which is defined differently in manual vs auto mode.
In manual mode this is simply a pointer, while in auto mode it's a vector type. See the details in `include/pto/common/memory.hpp`.

You should always keep this in mind to avoid using the returned value of `.data()` function directly as a pointer type outside tile functions.

# 2 - Avoid default initializer for a struct/class member

It's a very common practice to default-initialize data members in a struct or class in C++, for instance:

```cpp
struct ConvTile {
public:
    ...

    int shape[ConvTileDetail::MAX_CONVTILE_DIM] = {1};
};
```

This turns out to cause problems for the SROA pass in the compiler (SROA can't eliminate the `AllocaInst` of the struct plus all the load and store instructions associated with it). At least in auto mode, please DON'T default initialize the members:

```cpp
#ifdef __PTO_AUTO__
    // In auto mode, do not have default initialization in the class definition itself for its members
    int shape[ConvTileDetail::MAX_CONVTILE_DIM];
#else
    int shape[ConvTileDetail::MAX_CONVTILE_DIM] = {1};
#endif
```

Even though we are programming in C++, we encourage to use POD (Plain Old Data) Aggregate programming to describe structs and classes that is compatible with the C-programming language.

# 3 - Explicit synchronization is still needed inside tile functions and their callees

TL;DR:

- Use `set_flag`, `wait_flag` or `pipe_barrier` explicitly in tile functions and all of their callees.
- Use `PtoSetWaitFlag` or `TSYNC` anywhere else.

Reason:
The auto-sync will NOT traverse inside tile functions; as a matter of fact, the whole auto mode compiler works on the tile function level, meaning that everything inside tile function is a complete black box to auto-mode.

For this reason, if any synchronization is needed inside tile function, the library developers should still add synchronizations manually. That's why using `PtoSetWaitFlag` and `TSYNC` won't work in auto mode because it's no-op. Most of the cases this interface is used by kernel developers.

# 4 - Avoid using `TASSIGN` for implementation

Currently implementations of some pto instructions directly use `TASSIGN_IMPL`. This may be a problem for auto mode because it's no-op.

If you use `TASSIGN` just to alias 2 tiles, you should use `TRESHAPE` or `TSUBVIEW` to achieve the same goal depending on your needs. Anything else won't work for auto mode.

For instance, if you call `TASSIGN` to allocate memory based on some kind of algorithm, this will never work for auto-mode because the compiler can't possibly recognize the specific algorithm logic and do the same allocation as you want to do in manual mode.

After all, the whole memory allocation in auto mode is based on each individual tile's liveness analysis, without knowing any other context. This is why the current implementation of `TPUSH` and `TPOP` won't work for auto mode.

# 5 - Some general rules for `*_IMPL` functions

Some consistency must be ensured for `*_IMPL` and tile function interface:

- The function signature must have `PTO_INTERNAL` macro
- Its implementation should directly call tile functions inside, don't call any non-tile functions unless they're inlined.
- Always call `.data()` function to pass into tile functions, or return-by-reference for all return values of `.data()`. For example:

```cpp
TExp(dstTile.data(), srcTile.data()); // correct

auto dst = dstTile.data(); // wrong: return by value
auto &src = srcTile.data(); // correct: return by reference
TExp(dst, src);
```

# 6 - Some general rules for tile functions

- Ensure to use `typename <...>::TileDType` instead of `typename <...>::DType *` for tile types on tile function parameters
- Ensure these `typename <...>::TileDType` parameters are pass-by-value, not by pointer or reference
- Ensure `__in__` or `__out__` attributes are properly attached to these `typename <...>::TileDType` parameters
- Always call `__cce_get_tile_ptr` on these `typename <...>::TileDType` arguments to get a tile's underlying buffer pointer
- The return type should always be `void`. Otherwise the compiler's assumption about TF interface is broken and it's an undefined behavior. Please make all return values as pass-by-value arguments, even just for a single scalar.

# 7 - Avoid having runtime control flow before tile functions

Having runtime control flows imposes great challenges for auto-sync to work properly. We encourage developers to either try to remove these runtime conditions if they're not necessary or move them inside tile functions if possible.

Some examples include `TROWEXPANDDIV_IMPL` and `TMULS_IMPL`.
