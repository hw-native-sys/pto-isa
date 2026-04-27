# References and Further Reading

This document provides PTO development-related references, academic papers, online resources, and further reading to help developers deepen their understanding of PTO programming.

## Contents

- [Official Documentation](#official-documentation)
- [Example Code](#example-code)
- [Academic Papers](#academic-papers)
- [Online Resources](#online-resources)
- [Related Projects](#related-projects)
- [Tools and Libraries](#tools-and-libraries)
- [Recommended Books](#recommended-books)

---

## Official Documentation

### PTO-ISA Core Documentation

- **[PTO Virtual ISA Manual](../PTO-Virtual-ISA-Manual.md)**
  - Complete PTO instruction set architecture specification
  - Hardware abstraction model
  - Programming model details

- **[ISA Instruction Reference](../isa/scalar/ops/micro-instruction/README.md)**
  - Detailed description of all PTO instructions
  - Instruction syntax and semantics
  - Usage examples

- **[Programming Guide](README.md)**
  - PTO programming introduction
  - Best practices
  - Common patterns

### Topic-Specific Documentation

- **[Getting Started](../getting-started.md)**
  - Environment setup
  - First PTO program
  - Basic concepts

- **[Debugging Guide](debug.md)**
  - Debugging techniques
  - Troubleshooting common issues
  - Performance analysis

- **[Performance Optimization Guide](opt.md)**
  - Performance optimization strategies
  - Bottleneck analysis
  - Optimization cases

- **[Memory Optimization](memory-optimization.md)**
  - Memory management
  - Double buffering techniques
  - Memory alignment

- **[Pipeline and Parallel Execution](pipeline-parallel.md)**
  - Pipeline design
  - Multi-core parallelism
  - Event synchronization

- **[Operator Fusion](operator-fusion.md)**
  - Fusion patterns
  - Fusion implementation
  - Performance benefits

- **[Compilation Process](compilation-process.md)**
  - Compilation steps
  - Compilation options
  - Cross compilation

- **[Framework Integration](framework-integration.md)**
  - PyTorch integration
  - TensorFlow integration
  - ONNX Runtime integration

- **[Error Codes Reference](error-codes.md)**
  - Error code list
  - Solutions
  - Debugging tips

- **[Version Compatibility](version-compatibility.md)**
  - Version strategy
  - Platform compatibility
  - Migration guide

### CANN Documentation

- **[CANN Official Documentation](https://www.hiascend.com/document)**
  - CANN development guide
  - API reference
  - Tool usage

- **[AscendC Programming Guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha002/devguide/moddevg/ascendc/ascendc_0001.html)**
  - AscendC language reference
  - Operator development
  - Performance optimization

---

## Example Code

### Basic Examples

- **[Add Operator](../../demos/baseline/add/README.md)**
  - Simple element-wise addition
  - Basic Tile operations
  - Multi-core parallelism

- **[GEMM Basic](../../demos/baseline/gemm_basic/README.md)**
  - Matrix multiplication baseline
  - Tile blocking fundamentals
  - Build and test workflow

- **[Flash Attention Baseline](../../demos/baseline/flash_atten/README.md)**
  - Attention-style kernel baseline
  - Memory movement patterns
  - End-to-end demo workflow

### Advanced Examples

- **[GEMM Optimization](../../kernels/manual/a2a3/gemm_performance/README.md)**
  - Matrix multiplication optimization
  - Tiling strategies
  - Pipeline optimization
  - Performance tuning

- **[Flash Attention](../../kernels/manual/common/flash_atten/README.md)**
  - Attention mechanism implementation
  - Memory-efficient algorithm
  - Operator fusion

- **[TopK](../../kernels/manual/a2a3/topk/README.md)**
  - Selection and sorting flow
  - Reduction and data movement
  - Manual kernel structure

### Custom Operator Examples

- **[Fused Add-ReLU-Mul](../../kernels/custom/fused_add_relu_mul/README.md)**
  - Operator fusion example
  - Three implementation versions
  - Progressive optimization

---

## Academic Papers

### Tensor Compilers and DSLs

- **TVM: An Automated End-to-End Optimizing Compiler for Deep Learning**
  - Chen et al., OSDI 2018
  - Tensor compiler framework
  - Automatic optimization

- **Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation**
  - Ragan-Kelley et al., PLDI 2013
  - Image processing DSL
  - Schedule separation

- **Tiramisu: A Polyhedral Compiler for Expressing Fast and Portable Code**
  - Baghdadi et al., CGO 2019
  - Polyhedral model
  - Code generation

### Hardware Accelerators

- **In-Datacenter Performance Analysis of a Tensor Processing Unit**
  - Jouppi et al., ISCA 2017
  - Google TPU architecture
  - Performance analysis

- **NVIDIA A100 Tensor Core GPU: Performance and Innovation**
  - NVIDIA, 2020
  - GPU architecture
  - Tensor Core design

### Optimization Techniques

- **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**
  - Dao et al., NeurIPS 2022
  - Memory-efficient attention
  - Tiling strategy

- **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning**
  - Dao, 2023
  - Improved parallelism
  - Work partitioning

---

## Online Resources

### Official Websites

- **[Ascend Community](https://www.hiascend.com/)**
  - Official Ascend platform
  - Documentation and downloads
  - Community forum

- **[CANN GitHub](https://github.com/Ascend/cann)**
  - CANN source code
  - Issue tracking
  - Contribution guide

### Tutorials and Blogs

- **[Ascend Developer Blog](https://www.hiascend.com/blog)**
  - Technical articles
  - Best practices
  - Case studies

- **[PTO-ISA Examples Repository](../../demos/README.md)**
  - Hands-on examples
  - Step-by-step tutorials
  - Performance benchmarks

### Community Forums

- **[Ascend Forum](https://www.hiascend.com/forum)**
  - Q&A
  - Technical discussions
  - Community support

- **[GitHub Issues](https://github.com/PTO-ISA/pto-isa/issues)**
  - Bug reports
  - Feature requests
  - Technical discussions

---

## Related Projects

### Tensor Compilers

- **[TVM](https://tvm.apache.org/)**
  - Open-source tensor compiler
  - Multi-backend support
  - Auto-tuning

- **[XLA (Accelerated Linear Algebra)](https://www.tensorflow.org/xla)**
  - TensorFlow compiler
  - JIT compilation
  - Optimization

- **[MLIR (Multi-Level Intermediate Representation)](https://mlir.llvm.org/)**
  - Compiler infrastructure
  - Extensible IR
  - Reusable passes

### Deep Learning Frameworks

- **[PyTorch](https://pytorch.org/)**
  - Dynamic computation graphs
  - Python-first design
  - Rich ecosystem

- **[TensorFlow](https://www.tensorflow.org/)**
  - Production-ready
  - Multi-platform support
  - Comprehensive tools

- **[MindSpore](https://www.mindspore.cn/)**
  - Huawei AI framework
  - Native Ascend support
  - Auto-parallelism

### Performance Tools

- **[msprof](https://www.hiascend.com/document)**
  - Ascend profiler
  - Performance analysis
  - Bottleneck identification

- **[NVIDIA Nsight](https://developer.nvidia.com/nsight-systems)**
  - GPU profiler
  - System-wide analysis
  - Visualization

---

## Tools and Libraries

### Development Tools

- **CMake** (>= 3.16)
  - Build system generator
  - Cross-platform support
  - [cmake.org](https://cmake.org/)

- **GCC** (>= 13.0) / **Clang** (>= 15.0)
  - C++20 compiler
  - Optimization support
  - [gcc.gnu.org](https://gcc.gnu.org/)

- **Python** (>= 3.8)
  - Scripting and testing
  - Framework integration
  - [python.org](https://www.python.org/)

### Debugging Tools

- **GDB**
  - GNU debugger
  - Breakpoints and inspection
  - [gnu.org/software/gdb](https://www.gnu.org/software/gdb/)

- **Valgrind**
  - Memory error detection
  - Profiling
  - [valgrind.org](https://valgrind.org/)

### Performance Analysis

- **perf**
  - Linux profiler
  - Hardware counters
  - System-wide analysis

- **Intel VTune**
  - CPU profiler
  - Microarchitecture analysis
  - [intel.com/vtune](https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html)

---

## Recommended Books

### Computer Architecture

- **Computer Architecture: A Quantitative Approach**
  - Hennessy & Patterson
  - Classic architecture textbook
  - Performance analysis

- **Modern Processor Design: Fundamentals of Superscalar Processors**
  - Shen & Lipasti
  - Pipeline design
  - Instruction-level parallelism

### Parallel Programming

- **Programming Massively Parallel Processors**
  - Kirk & Hwu
  - GPU programming
  - CUDA fundamentals

- **Parallel Programming in C with MPI and OpenMP**
  - Quinn
  - Parallel patterns
  - Performance optimization

### Compiler Design

- **Engineering a Compiler**
  - Cooper & Torczon
  - Compiler construction
  - Optimization techniques

- **Advanced Compiler Design and Implementation**
  - Muchnick
  - Advanced optimizations
  - Code generation

### Deep Learning Systems

- **Deep Learning Systems: Algorithms, Compilers, and Processors**
  - Sze et al.
  - DL accelerators
  - System design

---

## Contributing

We welcome contributions to improve this documentation:

- **Report Issues**: [GitHub Issues](https://github.com/PTO-ISA/pto-isa/issues)
- **Submit PRs**: [GitHub Pull Requests](https://github.com/PTO-ISA/pto-isa/pulls)
- **Join Discussions**: [Ascend Forum](https://www.hiascend.com/forum)

---

## License

This documentation is licensed under [Apache License 2.0](https://github.com/PTO-ISA/pto-isa/blob/main/LICENSE).

---

## Contact

- **Email**: support@ascend.com
- **Forum**: [Ascend Community Forum](https://www.hiascend.com/forum)
- **GitHub**: [PTO-ISA Repository](https://github.com/PTO-ISA/pto-isa)
