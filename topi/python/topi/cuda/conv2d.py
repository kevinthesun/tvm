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
from tvm.autotvm.task.topi_integration import deserialize_args
from tvm.autotvm.task import get_config
from tvm.contrib import cudnn

from ..nn.conv2d import conv2d_NCHWc, conv2d_alter_layout
from .. import nn, generic, tag
from ..util import get_const_tuple, traverse_inline
from ..nn.pad import pad

from .conv2d_direct import schedule_direct_cuda, schedule_direct_conv2d_NCHWc_cuda
from .conv2d_winograd import winograd_cuda, schedule_winograd_cuda
from .conv2d_int8 import conv2d_NCHWc_int8, schedule_conv2d_NCHWc_int8


@autotvm.register_topi_compute(nn.conv2d, ['cuda', 'gpu'], ['direct', 'winograd', 'int8', 'NCHWc'])
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

    if layout == 'NCHW':
        return nn.conv2d_nchw(data, kernel, strides, padding, dilation, out_dtype)
    if layout == 'HWCN':
        return nn.conv2d_hwcn(data, kernel, strides, padding, dilation, out_dtype)
    raise ValueError("not support this layout {} yet".format(layout))


@autotvm.register_topi_schedule(generic.schedule_conv2d_nchw, ["cuda", "gpu"],
                                ["direct", 'winograd', "int8", "NCHWc"])
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
            schedule_direct_conv2d_NCHWc_cuda(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s


def _pack_data(cfg, data, kernel):
    oc, ic, kh, kw = get_const_tuple(kernel.shape)
    cfg.define_split("tile_ic", ic, num_outputs=2, filter=lambda y: 4 <= y.size[-1] <= 16)
    cfg.define_split("tile_oc", oc, num_outputs=2, filter=lambda y: 4 <= y.size[-1] <= 16)
    ic_bn, oc_bn = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1]

    n, _, ih, iw = get_const_tuple(data.shape)
    ic_chunk = ic // ic_bn
    oc_chunk = oc // oc_bn

    data = tvm.compute((n, ic_chunk, ih, iw, ic_bn),
                       lambda bs, c, h, w, vc: data[bs, c*ic_bn + vc, h, w],
                       name="packed_data")

    kernel = tvm.compute(
        (oc_chunk, ic_chunk, kh, kw, oc_bn, ic_bn),
        lambda occ, icc, k_h, k_w, ocb, icb:
        kernel[occ * oc_bn + ocb,
               icc * ic_bn + icb, k_h, k_w],
        name="packed_kernel")

    return data, kernel


#@autotvm.register_topi_compute(conv2d_NCHWc, ['cuda', 'gpu'], ['direct'])
def conv2d_NCHWc_cuda(cfg, data, kernel, strides, padding, dilation, layout, out_layout, out_dtype):
    # layout and out_layout are not used here,
    # we keep them for debug convenience when dumping autotvm workload
    HPAD, WPAD = padding if isinstance(padding, (tuple, list)) else (padding, padding)
    HSTR, WSTR = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    dh, dw = dilation if isinstance(dilation, (tuple, list)) else (dilation, dilation)

    oc_chunk, ic_chunk, kh, kw, oc_bn, ic_bn = get_const_tuple(kernel.shape)

    n, _, ih, iw, _ = get_const_tuple(data.shape)
    dilated_kernel_h = (kw - 1) * dh + 1
    dilated_kernel_w = (kh - 1) * dw + 1

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
