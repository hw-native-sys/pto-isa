# Communication ISA

Inter-NPU collective communication, point-to-point exchange, and runtime synchronization.

| | Instruction | PTO Name | Description |
|-|-----------|---------|-------------|
| | [TBROADCAST](./TBROADCAST.md) | `pto.tbroadcast` | Broadcast data from root NPU to all ranks |
| | [TGET](./TGET.md) | `pto.tget` | Get data from a remote NPU |
| | [TGET_ASYNC](./TGET_ASYNC.md) | `pto.tget_async` | Asynchronous variant of TGET |
| | [TNOTIFY](./TNOTIFY.md) | `pto.tnotify` | Notify other ranks of an event |
| | [TPUT](./TPUT.md) | `pto.tput` | Put data to a remote NPU |
| | [TPUT_ASYNC](./TPUT_ASYNC.md) | `pto.tput_async` | Asynchronous variant of TPUT |
| | [TREDUCE](./TREDUCE.md) | `pto.treduce` | Collective reduction across all ranks |
| | [TSCATTER](./TSCATTER.md) | `pto.tscatter` | Scatter data from root NPU to all ranks |
| | [TGATHER](./TGATHER.md) | `pto.tgather` | Gather data from all ranks to root NPU |
| | [TTEST](./TTEST.md) | `pto.ttest` | Test if a notification has been received |
| | [TWAIT](./TWAIT.md) | `pto.twait` | Wait for a notification |

See [Communication and Runtime](communication-runtime.md) for the instruction set contract.
