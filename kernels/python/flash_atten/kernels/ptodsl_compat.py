"""
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License.

Compatibility helpers for the validated ptodsl 0.1.2 API.

ptodsl 0.1.2 exposes pto.as_tensor() and pto.slice_view() as shape-dynamic
helpers that infer result types. This flash-attention builder needs explicit
TensorType/SubTensorType results so ptoas can preserve concrete slot and tile
shapes. Keep the private MLIR calls isolated here so the kernel builder does not
spread version-specific internals across the implementation.
"""

from mlir.dialects import pto as _mlir_pto
from ptodsl import pto, to_ir_module
from ptodsl.api.scalar import _unwrap
from ptodsl.api.type_def import _materialize
from ptodsl.utils.codegen import with_loc

_ORIG_AS_TENSOR = pto.as_tensor
_ORIG_SLICE_VIEW = pto.slice_view


def _compat_layout_attr(layout):
    if layout is None:
        return None
    if isinstance(layout, str):
        return _mlir_pto.LayoutAttr.get(getattr(_mlir_pto.Layout, layout))
    return layout


@with_loc
def as_tensor(tensor_type=None, *, ptr, shape, strides, layout=None):
    if tensor_type is None:
        return _ORIG_AS_TENSOR(ptr=ptr, shape=shape, strides=strides, layout=layout)
    kwargs = {}
    layout_attr = _compat_layout_attr(layout)
    if layout_attr is not None:
        kwargs["layout"] = layout_attr
    return _mlir_pto.MakeTensorViewOp(
        _materialize(tensor_type), _unwrap(ptr), [_unwrap(v) for v in shape], [_unwrap(v) for v in strides], **kwargs
    ).result


@with_loc
def slice_view(subtensor_type=None, *, source, offsets, sizes):
    if subtensor_type is None:
        return _ORIG_SLICE_VIEW(source=source, offsets=offsets, sizes=sizes)
    return _mlir_pto.PartitionViewOp(
        _materialize(subtensor_type),
        _unwrap(source),
        offsets=[_unwrap(v) for v in offsets],
        sizes=[_unwrap(v) for v in sizes],
    ).result


def to_ir_module_with_meta(*, meta_data, module=False):
    def decorator(fn):
        globals_to_update = fn.__globals__
        globals_to_update.update(meta_data())
        return to_ir_module(module=module)(fn)

    return decorator
