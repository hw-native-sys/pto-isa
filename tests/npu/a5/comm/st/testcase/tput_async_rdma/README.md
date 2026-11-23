# RDMA异步ST（HNS1825目标网卡平台）

本目录包含RDMA异步ST的共享实现。`tput_async_rdma` 验证远程WRITE，`tget_async_rdma` 验证远程READ，
`tput_async_notify_rdma` 验证带`Set`或`AtomicAdd`通知的远程WRITE；三个目标复用同一套RDMA测试Kernel实现。

## 前置条件

- 配备HNS1825网卡及匹配驱动、HCOMM组件的A5环境
- 按顶层[测试说明](../../../../../../README.md)配置MPI和HCCL
- 各参与rank使用同一地址族，RDMA网卡IPv4或IPv6互相可达

## 构建与运行

在CMake配置前选择RDMA实现，再运行所需的RDMA ST target：

```bash
export PTO_RDMA_BACKEND=HNS_1825
python3 tests/script/run_st.py -r npu -v a5 -t comm/tput_async_rdma -d -n 2
python3 tests/script/run_st.py -r npu -v a5 -t comm/tget_async_rdma -d -n 2
python3 tests/script/run_st.py -r npu -v a5 -t comm/tput_async_notify_rdma -d -n 2
```

`TPutAsyncRdma.Vec_Int32_MultiAiv` 验证16个AIV共享同一peer/QP时的PUT/notify、队列回绕及数据/Event正确性。
在 `tput_async_rdma` 命令中加 `-g TPutAsyncRdma.Vec_Int32_MultiAiv` 可单独运行。

多QP用例检查数据、独立完成及Event路由，包含16个AIV共享4个QP（每QP 4个AIV）的PUT/Notify Set/ADD和GET。
在上述PUT、GET命令中分别加 `-g 'TPutAsyncRdma.*MultiQp*'`、`-g 'TGetAsyncRdma.*MultiQp*'` 运行。

首次验证`TPUT_ASYNC_NOTIFY`时，先只运行2个rank的定向用例：

```bash
export PTO_RDMA_BACKEND=HNS_1825
python3 tests/script/run_st.py -r npu -v a5 -t comm/tput_async_notify_rdma \
    -g TPutAsyncNotifyRdma.Int32SetAndCanaries -d -n 2
```

该用例检查远端payload、远端signal及signal两侧canary，并等待接口返回的`AsyncEvent`完成。接收端观察到
signal后先维护数据缓存，再校验payload。首次验证ADD使用 `-g TPutAsyncNotifyRdma.Int32AddAndCanaries`，
signal从非零初值开始累加。`-g 'TPutAsyncNotifyRdma.Int32Add*'` 还覆盖负增量、零增量，以及4个AIV写入
互不重叠的payload并更新同一个signal。零增量不会改变signal，payload在完成后由Host校验。

`PTO_RDMA_BACKEND` 仅在CMake配置该ST构建时读取。未设置、空值或不支持的值会构建不含RDMA支持的测试。`run_st.py` 默认重新构建；修改该变量后，不要使用 `-w/--without-build` 复用已有二进制。

## 端点发现

Bootstrap先解析每个rank的物理设备id和RDMA IP，再通过MPI交换端点与注册内存信息。本地IP的查找顺序为：

1. 显式 `PTO_ROCE_LOCAL_IP`，其次是按rank排列的 `PTO_ROCE_IPS`。
2. `/etc/hccl_rootinfo.json` 中与物理设备匹配的RoCE地址。
3. HCOMM公共拓扑接口 `TopoAddrInfoGetSize` / `TopoAddrInfoGet`。

ST不会生成或修改拓扑文件。自动IPv6发现依赖HCOMM拓扑支持，发现失败时可显式配置IP。

| 变量 | 说明 |
|---|---|
| `PTO_RDMA_BACKEND` | 配置阶段选择项，当前唯一支持值为 `HNS_1825`。|
| `PTO_ROCE_ADDR_FAMILY` | RoCE地址族：`auto`（默认）、`inet4` 或 `inet6`。|
| `PTO_ROCE_PHYIDS` | 可选，按MPI rank索引、逗号分隔的物理设备id。|
| `PTO_ROCE_LOCAL_IP` | 当前MPI进程的显式IPv4/IPv6，优先于拓扑发现。|
| `PTO_ROCE_IPS` | 显式IPv4/IPv6列表，按MPI rank排列，数量必须等于rank数。|
| `PTO_ROCE_BASE_PORT` | 各rank一致的channel base port，默认 `60032`。|
| `PTO_ROCE_VERBOSE` | 设为 `1`，打印端点、MR、channel和释放进度。|
| `PTO_ISA_GLOBAL_LOG_LEVEL` | `0` DEBUG、`1` INFO、`2` WARNING、`3` ERROR（默认）；`4` 关闭PTO文件/stdout业务日志。|
| `PTO_ISA_SLOG_PRINT_TO_STDOUT` | 设为 `1` 时库日志输出到stdout，否则写入plog目录下的PTO文件。|
| `HCCL_RDMA_TC` | HCOMM traffic class，默认 `132`。|
| `HCCL_RDMA_SL` | HCOMM service level，默认 `4`。|

`PTO_ROCE_LOCAL_IP` 的优先级高于 `PTO_ROCE_IPS`。所有rank必须使用相同的地址族和base port；使用 `PTO_ROCE_IPS` 时，还必须使用相同的按rank排序列表。

仅用IPv6时，所有rank设置 `export PTO_ROCE_ADDR_FAMILY=inet6`。自动发现默认优先IPv4，再选IPv6；指定地址族后不回退。该变量仅控制RoCE。IPv6地址不带方括号、端口或scope后缀；不支持link-local IPv6，IPv4-mapped IPv6按IPv4处理。

## 问题定位

- CMake提示RDMA未使能时，设置 `PTO_RDMA_BACKEND=HNS_1825`，并在不使用 `-w` 的情况下重新配置。
- 端点发现失败时，检查物理设备映射以及root-info或virtual topology中的RoCE IPv4/IPv6，必要时显式配置IP。
- HCOMM无法从默认路径加载HNS1825 verbs provider时，将 `IBV_EXTEND_DRIVERS` 指向驱动提供的 `libhrn5-rdmav34.so`。
- 设置 `PTO_ROCE_VERBOSE=1`、`PTO_ISA_GLOBAL_LOG_LEVEL=1` 和 `PTO_ISA_SLOG_PRINT_TO_STDOUT=1`，在终端查看 INFO 级别的端点、MR、channel 和释放日志。
- 日志路径和文件管理规则见[Host日志](../../../../../../../include/pto/comm/logger/README.md)。

日志中的 `family` 和 `netdev` 表示所选地址族和网卡，`connect DONE` 表示RDMA初始化完成。
