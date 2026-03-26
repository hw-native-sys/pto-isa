# Auto Mode Overview

## Scope

This section gives an overview of PTO auto mode.

## What is Auto Mode

Auto mode is a compilation mode for PTO. It does all the memory allocation for tiles and synchronization in the compiler. Programming in auto mode works just like in manual mode, except there is no need for `TASSIGN` and `TSYNC`/`Event` (in fact, these will do nothing in auto mode).

## Why Use Auto Mode?

 The objective of auto mode is to provide users a higher level interface abstraction to improve productivity, while ensuring competitive performance compared to highly tuned code by an expert.

These abstractions include:

* Automatic synchronization instruction insertion between pipelines in the Ascend hardware
* Memory allocation and management for the `Tile` abstraction

## Auto Mode Features

The following sections will explain the features that auto mode offers such that it enables users to write in such a programming model as an alternative.

IMPORTANT NOTE: the standard layers of abstraction for all PTO instructions are expected to look like this (from highest to lowest):

* User API (public, highest-level interfaces called by kernel developers)
* IMPL layer API
* TF (Tile Function) layer API
* Internal CCE implementation API (e.g., VF, SIMT function, etc.)

The PTO compiler works on the abstraction level of "Tiles". That means, **all the features listed below only works above TF level**, because TF interface is the last level of abstraction for Tiles; once it gets inside tile function, it no longer has tile-level abstraction, only raw pointers with raw CCE intrinsics which is the realm of CCE. Therefore, the tile function is a complete black-box to PTO compiler; the PTO compiler will NOT work inside tile functions.

**Specifically, this means for all features listed below, they only work above tile function level; they won't work inside tile functions and on.**

### Automated Tile Live Range Analysis

In auto mode, PTO compilation will keep track of each Tile and its live-ranges. This is a core component that is used as analysis to serve the following features mentioned below.

### Automatic Synchronization

In manual mode, user would normally have to keep track of the asynchronous nature of the hardware by using PTO's [`event model`](../coding/Event.md) at precise code locations in order to ensure both functional correctness and high performance in execution. This might be tedious and error prone.

Auto mode compilation will allow users to avoid having to use the event model to synchronize their code. The compiler will automatically determine the locations to insert synchronization under the hood - ensuring functional correctness and competitive performance.

### Tile Memory Allocation

In the default mode of PTO compilation, after instantiating `Tile` variables, we would need to complement them with a `TASSIGN` instruction to manually assign a dedicated buffer address that it operates on. However in auto mode, this is not required anymore. By simply instantiating the `Tile` variable the compiler will automatically allocate the buffer addresses under the hood for the user.

## Compiling Auto Mode with Ascend CANN

To compile your kernel using PTO auto mode, all you need to do is enable the PTO auto mode passes in the Bisheng CCE toolchain using `--cce-enable-pto-passes`.

### Example (device compilation)

To compile a single CCE kernel source into an object file, make sure to adjust:

* `--cce-aicore-arch=...` for your SoC (examples in this repo use `dav-c220-vec`, `dav-c310-vec`, etc.).

For example:

```bash
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash

bisheng -c -x cce -O2 --cce-aicore-only \
  --cce-aicore-arch=dav-c310-vec \
  -std=c++17 \
  --cce-enable-pto-passes \
  kernel.cpp -o kernel.o
```
