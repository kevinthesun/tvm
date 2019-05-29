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

from ..nn.conv2d import conv2d_NCHWc, conv2d_alter_layout
from .. import nn, generic
from ..util import get_const_tuple, traverse_inline

from .conv2d_direct import schedule_direct_cuda, schedule_direct_conv2d_NCHWc_cuda
from .conv2d_winograd import winograd_cuda, schedule_winograd_cuda
from .conv2d_int8 import conv2d_NCHWc_int8, schedule_conv2d_NCHWc_int8


@autotvm.register_topi_compute(nn.conv2d, ['cuda', 'gpu'], ['direct', 'winograd', 'int8'])
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

    if layout == 'NCHW':
        return nn.conv2d_nchw(data, kernel, strides, padding, dilation, out_dtype)
    if layout == 'HWCN':
        return nn.conv2d_hwcn(data, kernel, strides, padding, dilation, out_dtype)
    raise ValueError("not support this layout {} yet".format(layout))


@autotvm.register_topi_schedule(generic.schedule_conv2d_nchw, ["cuda", "gpu"],
                                ["direct", 'winograd', "int8"])
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

    traverse_inline(s, outs[0].op, _callback)
    return s


@conv2d_alter_layout.register("cuda")
def _alter_conv2d_layout(attrs, inputs, tinfo, F):

    copy_inputs = [s for s in inputs]
    new_attrs = {k : attrs[k] for k in attrs.keys()}

    if F.__name__ == 'tvm.relay.op':
        # Derive channels for frontends (e.g ONNX) that miss "channel" field.
        new_attrs["channels"] = inputs[1].checked_type.shape[attrs['kernel_layout'].index('O')]

    data, kernel = tinfo[0], tinfo[1]
    batch_size, in_channel, height, width = get_const_tuple(data.shape)

    groups = attrs.get_int("groups")
    out_channel = attrs.get_int("channels") \
        if F.__name__ == 'nnvm.symbol' else new_attrs["channels"]
    padding = attrs.get_int_tuple("padding")
    strides = attrs.get_int_tuple("strides")
    dilation = attrs.get_int_tuple("dilation")
    out_dtype = attrs["out_dtype"]

    layout_name = 'layout' if F.__name__ == 'nnvm.symbol' else 'data_layout'

    layout = attrs[layout_name]
    kh, kw = attrs.get_int_tuple("kernel_size")

    dtype = data.dtype
    out_dtype = dtype if out_dtype in ("same", "") else out_dtype
    is_depthwise = groups == in_channel and groups == out_channel

    # only optimize for NCHW
    if layout != 'NCHW':
        return None
    if groups != 1 and not is_depthwise:
        return None

    dispatch_ctx = autotvm.task.DispatchContext.current
    target = tvm.target.current_target()
    # query schedule and fallback if necessary
    workload = autotvm.task.args_to_workload(
        [data, kernel, strides, padding, dilation, out_dtype], depthwise_conv2d_nchw) \
        if is_depthwise else \
        autotvm.task.args_to_workload(
            [data, kernel, strides, padding, dilation, layout, out_dtype], conv2d)
    cfg = dispatch_ctx.query(target, workload)
    if cfg.is_fallback:
        _get_default_config(cfg, data, kernel, strides, padding, out_dtype, is_depthwise)

    ic_bn, oc_bn = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1]

    new_attrs[layout_name] = 'NCHW%dc' % ic_bn
    new_attrs['out_layout'] = 'NCHW%dc' % oc_bn

    new_data = tvm.placeholder((batch_size, in_channel//ic_bn, height, width, ic_bn),
                               dtype=data.dtype)

    if is_depthwise:
        raise RuntimeError("Not supported depthwise conv2d for cuda now.")
    else:
        out_channel, _, kh, kw = get_const_tuple(kernel.shape)
        # (oc, ic, h, w) -> (OC, IC, h, w, ic, oc)
        new_attrs['kernel_layout'] = 'OIHW%di%do' % (ic_bn, oc_bn)

        # Store altered operator's config
        new_kernel = tvm.placeholder((out_channel//oc_bn, in_channel//ic_bn, kh, kw, ic_bn, oc_bn),
                                     dtype=kernel.dtype)
        new_workload = autotvm.task.args_to_workload(
            [new_data, new_kernel, strides, padding, dilation, new_attrs[layout_name],
             new_attrs['out_layout'], out_dtype], conv2d_NCHWc)

    dispatch_ctx.update(target, new_workload, cfg)

    if is_depthwise:
        if F.__name__ == 'nnvm.symbol':
            logging.warning("Use native layout for depthwise convolution on NNVM.")
            return None
        return F.nn.contrib_depthwise_conv2d_nchwc(*copy_inputs, **new_attrs)
    else:
        if F.__name__ == 'nnvm.symbol':
            return F.contrib.conv2d_NCHWc(*copy_inputs, **new_attrs)
        return F.nn.contrib_conv2d_nchwc(*copy_inputs, **new_attrs)


@autotvm.register_topi_compute(conv2d_NCHWc, ['cuda', 'gpu'], ['direct'])
def conv2d_NCHWc_cuda(cfg, data, kernel, strides,
                      padding, dilation, layout, out_layout, out_dtype):
    # layout and out_layout are not used here,
    # we keep them for debug convenience when dumping autotvm workload
    HPAD, WPAD = padding if isinstance(padding, (tuple, list)) else (padding, padding)
    HSTR, WSTR = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    dh, dw = dilation if isinstance(dilation, (tuple, list)) else (dilation, dilation)

    n, ic_chunk, ih, iw, ic_bn = get_const_tuple(data.shape)
    in_channel = ic_chunk * ic_bn
    oc_chunk, ic_chunk_group, kernel_height, kernel_width, _, oc_bn = \
        get_const_tuple(kernel.shape)
    dilated_kernel_h = (kernel_height - 1) * dh + 1
    dilated_kernel_w = (kernel_width - 1) * dw + 1
    groups = ic_chunk // ic_chunk_group

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

    ic = tvm.reduce_axis((0, in_channel), name='ic')
    kh = tvm.reduce_axis((0, kernel_height), name='kh')
    kw = tvm.reduce_axis((0, kernel_width), name='kw')

    # else: fp implementation
    return tvm.compute(oshape, lambda n, oc_chunk, oh, ow, oc_block:
    tvm.sum(data_pad[n, ic//ic_bn, oh*HSTR+kh*dh, ow*WSTR+kw*dw,
                     ic%ic_bn].astype(out_dtype) *
            kernel[oc_chunk, ic//ic_bn, kh, kw, ic%ic_bn, oc_block],
            axis=[ic, kh, kw]),
                       name='conv2d_NCHWc', tag="conv2d_NCHWc")

@autotvm.register_topi_schedule(generic.schedule_conv2d_NCHWc, 'cpu', ['direct'])
def schedule_conv2d_NCHWc_cuda(cfg, outs):
    """Create schedule for tensors"""
    s = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []

    def traverse(op):
        """Traverse operators from computation graph"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if tensor.op.input_tensors and tensor.op not in scheduled_ops:
                    traverse(tensor.op)

        if 'conv2d_NCHWc' in op.tag:
            conv_out = op.output(0)
            kernel = conv_out.op.input_tensors[1]
            data_vec = conv_out.op.input_tensors[0]
            args = [s, cfg, data_vec, conv_out, outs[0]]
            _, _, kh, kw, _, _, = get_const_tuple(kernel.shape)
            schedule_conv2d_NCHWc_cuda(*args)

        scheduled_ops.append(op)

    traverse(outs[0].op)
    return s
