# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name
"""Compute definition for conv2d with cuda backend"""
import tvm
from tvm import autotvm
from tvm.contrib import cudnn

from ..nn.conv2d import conv2d_NCHWc, conv2d_alter_layout, conv2d_infer_layout
from ..nn import conv2d, group_conv2d_nchw, conv2d_winograd_without_weight_transform
from .. import nn, generic
from ..util import get_const_tuple, traverse_inline, get_dynamic_shape
from ..nn.pad import pad

from .conv2d_direct import schedule_direct_cuda, schedule_conv2d_nchwc_cuda
from .conv2d_winograd import winograd_cuda, winograd_NCHWc_cuda, schedule_winograd_cuda, \
    schedule_winograd_NCHWc_cuda, infer_tile_size
from .conv2d_int8 import conv2d_NCHWc_int8, schedule_conv2d_NCHWc_int8


@autotvm.register_topi_compute(nn.conv2d, ['cuda', 'gpu'], ['direct', 'winograd', 'int8',
                                                            'NCHWc', 'winograd_NCHWc'])
def conv2d_cuda(cfg, data, kernel, strides, padding, dilation, layout='NCHW', out_dtype='float32'):
    """Conv2D operator for cuda backend.

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template

    data : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width] or
        5-D with shape [batch, ic_chunk, in_height, in_width, ic_block]

    kernel : tvm.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width] or
        6-D with shape [num_filter_chunk, in_channel_chunk, filter_height,
        filter_width, num_filter_block, in_channel_block]

    strides : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    layout : str
        layout of data

    out_dtype: str
        The output type. This is used for mixed precision.

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    target = tvm.target.current_target()

    if "cudnn" in target.libs:
        if layout == 'NCHW':
            tensor_format = 0 # CUDNN_TENSOR_NCHW
            N, _, H, W = get_const_tuple(data.shape)
        elif layout == 'NHWC':
            tensor_format = 1 # CUDNN_TENSOR_NHWC
            N, H, W, _ = get_const_tuple(data.shape)
        else:
            raise ValueError("Unsupported layout %s in cudnn" % layout)
        CO, CI, KH, KW = get_const_tuple(kernel.shape)

        # handle dilation
        stride_h, stride_w = (strides, strides) if isinstance(strides, int) else strides
        pad_h, pad_w = (padding, padding) if isinstance(padding, int) else padding
        dilation_h, dilation_w = (dilation, dilation) if isinstance(dilation, int) else dilation

        OH = (H + 2 * pad_h - KH) // stride_h + 1
        OW = (W + 2 * pad_w - KW) // stride_w + 1
        cfg.add_flop(2 * N * OH * OW * CO * CI * ((KH - 1) * dilation_h + 1) *\
                    ((KW - 1) * dilation_w + 1))

        return cudnn.conv2d_forward(data,
                                    kernel,
                                    stride_h,
                                    stride_w,
                                    pad_h,
                                    pad_w,
                                    dilation_h,
                                    dilation_w,
                                    conv_mode=1,
                                    tensor_format=tensor_format,
                                    algo=-1)  # let CUDNN choose the best algo

    if cfg.template_key == 'winograd':
        return winograd_cuda(cfg, data, kernel, strides, padding, dilation, layout, out_dtype,
                             pre_computed=False)
    if cfg.template_key == 'int8':
        if (data.dtype == 'int8' or data.dtype == 'uint8'):
            return conv2d_NCHWc_int8(
                cfg, data, kernel, strides, padding, dilation, layout, out_dtype)

    if cfg.template_key == 'NCHWc':
        data, kernel = _pack_data(cfg, data, kernel)
        return conv2d_NCHWc_cuda(cfg, data, kernel, strides, padding, dilation, layout, None, out_dtype)

    if cfg.template_key == 'winograd_NCHWc':
        data, kernel = _pack_data(cfg, data, kernel, False)
        return winograd_NCHWc_cuda(cfg, data, kernel, strides, padding, dilation, layout, out_dtype, False)

    if layout == 'NCHW':
        return nn.conv2d_nchw(data, kernel, strides, padding, dilation, out_dtype)
    if layout == 'HWCN':
        return nn.conv2d_hwcn(data, kernel, strides, padding, dilation, out_dtype)
    raise ValueError("not support this layout {} yet".format(layout))


@autotvm.register_topi_schedule(generic.schedule_conv2d_nchw, ["cuda", "gpu"],
                                ["direct", 'winograd', "int8", "NCHWc", "winograd_NCHWc"])
def schedule_conv2d_nchw_cuda(cfg, outs):
    """TOPI schedule callback of conv2d for cuda gpu

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template

    outs: Array of Tensor
        The computation graph description of conv2d
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for conv2d.
    """
    target = tvm.target.current_target()
    if 'cudnn' in target.libs:
        return generic.schedule_extern(outs)

    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == 'conv2d_nchw':
            schedule_direct_cuda(cfg, s, op.output(0))
        if op.tag == 'conv2d_nchw_winograd':
            schedule_winograd_cuda(cfg, s, op.output(0), pre_computed=False)
        if op.tag == "conv2d_NCHWc_int8":
            schedule_conv2d_NCHWc_int8(cfg, s, op.output(0))
        if op.tag == "conv2d_NCHWc":
            schedule_conv2d_nchwc_cuda(cfg, s, op.output(0))
        if op.tag == "conv2d_NCHWc_winograd":
            schedule_winograd_NCHWc_cuda(cfg, s, op.output(0), False)

    traverse_inline(s, outs[0].op, _callback)
    return s


def _generate_split_factor(num_filters):
    num_candidates = 3
    tmp = []
    for i in range(num_filters):
        if num_filters % (i + 1) == 0:
            tmp.append(i + 1)
    ret = []
    for i in tmp:
        if len(ret) == num_candidates:
            break
        if 4 <= i <= 16:
            ret.append(i)
    if len(ret) < num_candidates:
        tmp.reverse()
        for i in tmp:
            if i not in ret:
                ret.append(i)
            if len(ret) == num_candidates:
                break
    return ret


def _pack_data(cfg, data, kernel, pack_kernel=True):
    n, _, ih, iw = get_const_tuple(data.shape)
    oc, ic, kh, kw = get_const_tuple(kernel.shape)

    ic_space = _generate_split_factor(ic)
    oc_space = _generate_split_factor(oc)
    cfg.define_knob("tile_ic", ic_space)
    cfg.define_knob("tile_oc", oc_space)
    ic_bn, oc_bn = cfg["tile_ic"].val, cfg["tile_oc"].val

    n, _, ih, iw = get_const_tuple(data.shape)
    ic_chunk = ic // ic_bn
    oc_chunk = oc // oc_bn

    data = tvm.compute((n, ic_chunk, ih, iw, ic_bn),
                       lambda bs, c, h, w, vc: data[bs, c*ic_bn + vc, h, w],
                       name="packed_data")

    if pack_kernel:
        kernel = tvm.compute(
            (oc_chunk, ic_chunk, kh, kw, oc_bn, ic_bn),
            lambda occ, icc, k_h, k_w, ocb, icb:
            kernel[occ * oc_bn + ocb,
                   icc * ic_bn + icb, k_h, k_w],
            name="packed_kernel")

    return data, kernel


@autotvm.register_topi_compute(conv2d_NCHWc, ['cuda', 'gpu'], ['NCHWc', 'winograd_NCHWc'])
def conv2d_NCHWc_cuda(cfg, data, kernel, strides, padding, dilation, layout, out_layout, out_dtype):
    """Conv2D operator for nChw[x]c layout.

    Parameters
    ----------
    data : tvm.Tensor
        5-D with shape [batch, in_channel_chunk, in_height, in_width, in_channel_block]

    kernel : tvm.Tensor
        6-D with shape
        [num_filter_chunk, in_channel_chunk, filter_height, filter_width,
        in_channel_block, num_filter_block] or
        4-D pre-computed with rely.nn.contrib_conv2d_winograd_weight_transform

    strides : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    layout : str
        Input data layout

    out_layout : str
        Output data layout

    out_dtype : str
        output data type

    Returns
    -------
    output : tvm.Tensor
        5-D with shape [batch, out_channel_chunk, out_height, out_width, out_channel_block]
    """
    if cfg.template_key == "NCHWc":
        return _conv2d_direct_NCHWc(cfg, data, kernel, strides, padding, dilation, layout, out_layout, out_dtype)
    if cfg.template_key == "winograd_NCHWc":
        return winograd_NCHWc_cuda(cfg, data, kernel, strides, padding, dilation, layout, out_dtype, True)


def _conv2d_direct_NCHWc(cfg, data, kernel, strides, padding, dilation, layout, out_layout, out_dtype):
    """Conv2d NCHWc for direct compute."""
    # layout and out_layout are not used here,
    # we keep them for debug convenience when dumping autotvm workload
    HPAD, WPAD = padding if isinstance(padding, (tuple, list)) else (padding, padding)
    HSTR, WSTR = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    dh, dw = dilation if isinstance(dilation, (tuple, list)) else (dilation, dilation)

    oc_chunk, ic_chunk, kh, kw, oc_bn, ic_bn = get_const_tuple(kernel.shape)

    dshape = get_const_tuple(data.shape)
    n, _, ih, iw, _ = get_dynamic_shape(dshape)
    dilated_kernel_h = (kh - 1) * dh + 1
    dilated_kernel_w = (kw - 1) * dw + 1

    # output shape
    out_height = (ih + 2 * HPAD - dilated_kernel_h) // HSTR + 1
    out_width = (iw + 2 * WPAD - dilated_kernel_w) // WSTR + 1
    oshape = (n, oc_chunk, out_height, out_width, oc_bn)

    # DOPAD
    DOPAD = (HPAD != 0 or WPAD != 0)
    if DOPAD:
        data_pad = pad(data, (0, 0, HPAD, WPAD, 0), name="data_pad")
    else:
        data_pad = data

    icc = tvm.reduce_axis((0, ic_chunk), name='ic_chunk')
    icb = tvm.reduce_axis((0, ic_bn), name='ic_block')
    kh = tvm.reduce_axis((0, kh), name='kh')
    kw = tvm.reduce_axis((0, kw), name='kw')

    return tvm.compute(oshape, lambda bs, occ, oh, ow, oc_block:
    tvm.sum(data_pad[bs, icc, oh*HSTR+kh*dh, ow*WSTR+kw*dw,
                     icb].astype(out_dtype) *
            kernel[occ, icc, kh, kw, oc_block, icb],
            axis=[icc, kh, kw, icb]),
                       name='conv2d_NCHWc', tag="conv2d_NCHWc")


@autotvm.register_topi_schedule(generic.schedule_conv2d_NCHWc, ["cuda", "gpu"],
                                ["NCHWc", "winograd_NCHWc"])
def schedule_conv2d_NCHWc_cuda(cfg, outs):
    """TOPI schedule callback of conv2d_NCHWc for cuda gpu

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template

    outs: Array of Tensor
        The computation graph description of conv2d
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for conv2d.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "conv2d_NCHWc":
            schedule_conv2d_nchwc_cuda(cfg, s, op.output(0))
        if op.tag == "conv2d_NCHWc_winograd":
            schedule_winograd_NCHWc_cuda(cfg, s, op.output(0), True)

    traverse_inline(s, outs[0].op, _callback)
    return s


@conv2d_infer_layout.register(["cuda", "gpu"])
def _conv2d_infer_layout(workload, cfg):
    _, data, kernel, strides, padding, dilation, layout, dtype = workload
    batch_size, in_channel, in_height, in_width = data[:-1]
    out_channel, _, k_height, k_width = kernel[:-1]
    out_height = (in_height + 2 * padding[0] - k_height) // strides[0] + 1
    out_width = (in_width + 2 * padding[1] - k_width) // strides[1] + 1

    if "NCHWc" in cfg.template_key:
        tile_ic, tile_oc = cfg["tile_ic"].val, cfg["tile_oc"].val
        in_shape = (batch_size, in_channel // tile_ic, in_height, in_width, tile_ic)
        in_layout = "NCHW%dc" % tile_ic
        out_shape = (batch_size, out_channel // tile_oc, out_height, out_width, tile_oc)
        out_layout = "NCHW%dc" % tile_oc
    else:
        in_shape = (batch_size, in_channel, in_height, in_width)
        out_shape = (batch_size, out_channel, out_height, out_width)
        in_layout = out_layout = layout

    return ((in_shape, in_layout),), ((out_shape, out_layout),)


@conv2d_alter_layout.register(["cuda", "gpu"])
def _alter_conv2d_layout(attrs, inputs, tinfos, F):
    """Alter op layout for pre-computing kernel transformation

    Parameters
    ----------
    attrs : nnvm.top.AttrDict or tvm.attrs.Attrs
        Attributes of current convolution
    inputs : nnvm.symbol or tvm.relay.Expr
        Grouped input symbols
    tinfos : list
        Input shape and dtype
    F: symbol
        The context, can be either nnvm.sym or relay.op

    Note
    ----
    Unlike other TOPI functions, this function operates on both graph level and operator level,
    so we have to pass 'F' to make it support our two versions of graph IR, NNVM and Relay.
    """
    if 'cudnn' in tvm.target.current_target().libs or 'miopen' in tvm.target.current_target().libs:
        return None

    copy_inputs = [s for s in inputs]
    new_attrs = {k: attrs[k] for k in attrs.keys()}

    if F.__name__ == 'tvm.relay.op':
        # Derive channels for frontends (e.g ONNX) that miss "channel" field.
        new_attrs["channels"] = inputs[1].checked_type.shape[attrs['kernel_layout'].index('O')]

    strides = attrs.get_int_tuple("strides")
    padding = attrs.get_int_tuple("padding")
    dilation = attrs.get_int_tuple("dilation")
    groups = attrs.get_int('groups')
    data_layout_key = "data_layout" if "data_layout" in new_attrs else "layout"
    layout = attrs[data_layout_key]
    out_dtype = attrs["out_dtype"]
    if out_dtype in ("", "same"):
        out_dtype = tinfos[0].dtype

    data, kernel = tinfos[0:2]
    N, CI, H, W = get_const_tuple(data.shape)
    if isinstance(N, tvm._ffi._ctypes.node.NodeBase):
        # This is a hack.
        N = 1#tvm.var("n")
    CO, _, KH, KW = get_const_tuple(kernel.shape)

    dispatch_ctx = autotvm.DispatchContext.current
    target = tvm.target.current_target()

    if groups == 1:
        # query config of this workload
        # This is hack again
        data = tvm.placeholder((N, CI, H, W), dtype=data.dtype)
        kernel = tvm.placeholder(kernel.shape, dtype=kernel.dtype)
        workload = autotvm.task.args_to_workload(
            [data, kernel, strides, padding, dilation, layout, out_dtype], conv2d)
        cfg = autotvm.DispatchContext.current.query(target, workload)

        if cfg.is_fallback:  # if is fallback, clear query cache and return None
            autotvm.task.clear_fallback_cache(target, workload)
            return None

        if cfg.template_key == 'direct':
            return None

        if cfg.template_key == "NCHWc":
            ic_bn, oc_bn = cfg["tile_ic"].val, cfg["tile_oc"].val
            new_attrs[data_layout_key] = 'NCHW%dc' % ic_bn
            new_attrs['out_layout'] = 'NCHW%dc' % oc_bn
            new_attrs['kernel_layout'] = 'OIHW%do%di' % (oc_bn, ic_bn)
            new_data = tvm.placeholder((N, CI // ic_bn, H, W, ic_bn),
                                       dtype=data.dtype)
            new_kernel = tvm.placeholder((CO // oc_bn, CI // ic_bn, KH, KW, oc_bn, ic_bn),
                                         dtype=kernel.dtype)
            new_workload = autotvm.task.args_to_workload(
                [new_data, new_kernel, strides, padding, dilation, new_attrs[data_layout_key],
                 new_attrs['out_layout'], out_dtype], conv2d_NCHWc)
            dispatch_ctx.update(target, new_workload, cfg)
            if F.__name__ == 'nnvm.symbol':
                return F.contrib.conv2d_NCHWc(*copy_inputs, **new_attrs)
            return F.nn.contrib_conv2d_nchwc(*copy_inputs, **new_attrs)

        if cfg.template_key == 'int8':
            assert 'cuda' in target.keys
            new_layout = 'NCHW4c'
            new_attrs[data_layout_key] = new_layout
            new_attrs['out_layout'] = new_layout
            new_attrs['kernel_layout'] = 'OIHW4o4i'
            ic_block_factor = oc_block_factor = 4

            # Store the same config for the altered operator (workload)
            new_data = tvm.placeholder((N, CI // ic_block_factor, H, W, ic_block_factor),
                                       dtype=data.dtype)
            new_kernel = tvm.placeholder((CO // oc_block_factor, CI // ic_block_factor, KH, KW, \
                                          oc_block_factor, ic_block_factor), dtype=kernel.dtype)
            new_workload = autotvm.task.args_to_workload(
                [new_data, new_kernel, strides, padding, dilation, new_layout, out_dtype],
                conv2d
            )
            dispatch_ctx.update(target, new_workload, cfg)
            return F.nn.conv2d(*copy_inputs, **new_attrs)

        if attrs.get_int_tuple("dilation") != (1, 1):
            warnings.warn("Does not support weight pre-transform for dilated convolution.")
            return None

        # pre-compute weight transformation in winograd
        tile_size = infer_tile_size(tinfos[0], tinfos[1])

        weight = F.nn.contrib_conv2d_winograd_weight_transform(copy_inputs[1],
                                                               tile_size=tile_size)
        weight = F.transpose(weight, axes=[0, 1, 3, 2])
        copy_inputs[1] = weight

        if cfg.template_key == "winograd_NCHWc":
            ic_bn, oc_bn = cfg["tile_ic"].val, cfg["tile_oc"].val
            new_attrs[data_layout_key] = 'NCHW%dc' % ic_bn
            new_attrs['out_layout'] = 'NCHW%dc' % oc_bn
            new_data = tvm.placeholder((N, CI // ic_bn, H, W, ic_bn), dtype=data.dtype)
            new_kernel = tvm.placeholder((KH + tile_size - 1, KW + tile_size - 1, CI, CO), 
                                         dtype=kernel.dtype)
            new_workload = autotvm.task.args_to_workload(
                [new_data, new_kernel, strides, padding, dilation, new_attrs[data_layout_key],
                 new_attrs['out_layout'], out_dtype], conv2d_NCHWc)
            dispatch_ctx.update(target, new_workload, cfg)
            if F.__name__ == 'nnvm.symbol':
                return F.contrib.conv2d_NCHWc(*copy_inputs, **new_attrs)
            return F.nn.contrib_conv2d_nchwc(*copy_inputs, **new_attrs)

        new_attrs['tile_size'] = tile_size

        # Store the same config for the altered operator (workload)
        new_data = data
        new_weight = tvm.placeholder((KH + tile_size - 1, KW + tile_size - 1, CI, CO),
                                     dtype=kernel.dtype)
        new_workload = autotvm.task.args_to_workload(
            [new_data, new_weight, strides, padding, dilation, layout, out_dtype, tile_size],
            conv2d_winograd_without_weight_transform
        )
        dispatch_ctx.update(target, new_workload, cfg)
        return F.nn.contrib_conv2d_winograd_without_weight_transform(*copy_inputs, **new_attrs)

    if groups != CI:
        workload = autotvm.task.args_to_workload(
            [tinfos[0], tinfos[1], strides, padding, dilation, groups, out_dtype],
            group_conv2d_nchw)
        cfg = autotvm.DispatchContext.current.query(target, workload)

        if cfg.is_fallback:  # if is fallback, clear query cache and return None
            autotvm.task.clear_fallback_cache(target, workload)
            return None

        if cfg.template_key == 'int8':
            assert 'cuda' in target.keys
            new_layout = 'NCHW4c'
            new_attrs[data_layout_key] = new_layout
            new_attrs['out_layout'] = new_layout
            new_attrs['kernel_layout'] = 'OIHW4o4i'
            ic_block_factor = oc_block_factor = 4

            # Store the same config for the altered operator (workload)
            new_data = tvm.placeholder((N, CI // ic_block_factor, H, W, ic_block_factor),
                                       dtype=data.dtype)
            new_kernel = tvm.placeholder((CO // oc_block_factor, CI // ic_block_factor // groups, \
                                          KH, KW, oc_block_factor, ic_block_factor),
                                         dtype=kernel.dtype)
            new_workload = autotvm.task.args_to_workload(
                [new_data, new_kernel, strides, padding, dilation, groups, out_dtype],
                group_conv2d_nchw
            )
            dispatch_ctx.update(target, new_workload, cfg)
            return F.nn.conv2d(*copy_inputs, **new_attrs)

    # do nothing for depthwise convolution
    return None
