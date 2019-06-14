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
# pylint: disable=invalid-name,unused-variable,unused-argument
"""Winograd template for cuda backend"""

import tvm
from tvm import autotvm

from .injective import _schedule_injective
from .. import nn
from ..nn import conv2d_winograd_without_weight_transform
from ..util import get_const_int, get_const_tuple, traverse_inline
from ..generic import schedule_conv2d_winograd_without_weight_transform
from ..nn.winograd_util import winograd_transform_matrices


def infer_tile_size(data, kernel):
    if len(data.shape) == 4:
        N, CI, H, W = get_const_tuple(data.shape)
    elif len(data.shape) == 5:
        N, CI_CHUNK, H, W, CI_BLOCK = get_const_tuple(data.shape)
    else:
        raise RuntimeError("Cuda conv2d winograd only supports 4-D or 5-D input data.")

    if H % 8 == 0:
        return 4
    return 2

def winograd_cuda(cfg, data, kernel, strides, padding, dilation, layout, out_dtype, pre_computed):
    """Compute declaration for winograd"""
    assert layout == 'NCHW'

    tile_size = infer_tile_size(data, kernel)

    N, CI, H, W = get_const_tuple(data.shape)

    if not pre_computed: # kernel tensor is raw tensor, do strict check
        if isinstance(dilation, int):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation
        if dilation_h != 1 or dilation_w != 1:
            kernel = nn.dilate(kernel, (1, 1, dilation_h, dilation_w))

        CO, CI, KH, KW = get_const_tuple(kernel.shape)
        HPAD, WPAD, _, _ = nn.get_pad_tuple(padding, kernel)
        HSTR, WSTR = (strides, strides) if isinstance(strides, int) else strides
        assert HSTR == 1 and WSTR == 1 and KH == KW
    else:                   # kernel tensor is pre-transfomred. this op is created by
                            # alter op layout, do not check
        # dilation is not supported
        HSTR = WSTR = 1
        HPAD = WPAD = 1
        KH = KW = 3
        _, _, CI, CO = get_const_tuple(kernel.shape)

    data_pad = nn.pad(data, (0, 0, HPAD, WPAD), (0, 0, HPAD, WPAD), name="data_pad")

    r = KW
    m = tile_size
    alpha = m + r - 1
    A, B, G = winograd_transform_matrices(m, r, out_dtype)

    H = (H + 2 * HPAD - KH) // HSTR + 1
    W = (W + 2 * WPAD - KW) // WSTR + 1
    nH, nW = (H + m-1) // m, (W + m-1) // m
    P = N * nH * nW

    # transform kernel
    if not pre_computed:
        r_kh = tvm.reduce_axis((0, KH), name='r_kh')
        r_kw = tvm.reduce_axis((0, KW), name='r_kw')
        kernel_pack = tvm.compute((alpha, alpha, CI, CO), lambda eps, nu, ci, co:
                                  tvm.sum(kernel[co][ci][r_kh][r_kw] *
                                          G[eps][r_kh] * G[nu][r_kw],
                                          axis=[r_kh, r_kw]), name='kernel_pack')
    else:
        kernel_pack = kernel

    # pack input tile
    input_tile = tvm.compute((CI, P, alpha, alpha), lambda c, p, eps, nu:
                             data_pad[p // (nH * nW)][c][p // nW % nH * m + eps]
                             [p % nW * m + nu], name='d')

    # transform data
    r_a = tvm.reduce_axis((0, alpha), 'r_a')
    r_b = tvm.reduce_axis((0, alpha), 'r_a')
    data_pack = tvm.compute((alpha, alpha, CI, P), lambda eps, nu, ci, p:
                            tvm.sum(input_tile[ci][p][r_a][r_b] * B[r_a][eps] * B[r_b][nu],
                                    axis=[r_a, r_b]), name='data_pack')

    # do batch gemm
    ci = tvm.reduce_axis((0, CI), name='ci')
    bgemm = tvm.compute((alpha, alpha, CO, P), lambda eps, nu, co, p:
                        tvm.sum(kernel_pack[eps][nu][ci][co] *
                                data_pack[eps][nu][ci][p],
                                axis=[ci]), name='bgemm')

    # inverse transform
    r_a = tvm.reduce_axis((0, alpha), 'r_a')
    r_b = tvm.reduce_axis((0, alpha), 'r_a')
    inverse = tvm.compute((CO, P, m, m), lambda co, p, vh, vw:
                          tvm.sum(bgemm[r_a][r_b][co][p] * A[r_a][vh] * A[r_b][vw],
                                  axis=[r_a, r_b]), name='inverse')

    # output
    output = tvm.compute((N, CO, H, W), lambda n, co, h, w:
                         inverse[co][n * nH * nW + (h // m) * nW + w // m][h % m][w % m],
                         name='output', tag='conv2d_nchw_winograd')
    cfg.add_flop(2 * N * CO * H * W * CI * KH * KW)

    return output


def schedule_winograd_cuda(cfg, s, output, pre_computed):
    """Schedule winograd template"""
    # get stages
    inverse = s[output].op.input_tensors[0]
    bgemm, A = s[inverse].op.input_tensors
    kernel_pack, data_pack = s[bgemm].op.input_tensors
    input_tile, B = s[data_pack].op.input_tensors
    pad_data = s[input_tile].op.input_tensors[0]

    # data transform
    s[B].compute_inline()

    data_l = s.cache_write(data_pack, 'local')
    eps, nu, c, p = s[data_l].op.axis
    r_a, r_b = s[data_l].op.reduce_axis
    for axis in [eps, nu, r_a, r_b]:
        s[data_l].unroll(axis)

    eps, nu, c, p = s[data_pack].op.axis
    p, pi = s[data_pack].split(p, 1)
    fused = s[data_pack].fuse(c, p)
    bb, tt = s[data_pack].split(fused, 128)
    s[data_pack].reorder(bb, tt, pi, eps, nu)
    s[data_pack].bind(bb, tvm.thread_axis("blockIdx.x"))
    s[data_pack].bind(tt, tvm.thread_axis("threadIdx.x"))

    s[data_l].compute_at(s[data_pack], pi)
    s[input_tile].compute_at(s[data_pack], pi)
    s[pad_data].compute_inline()

    # transform kernel
    if not pre_computed:
        kernel, G = s[kernel_pack].op.input_tensors
        eps, nu, ci, co = s[kernel_pack].op.axis
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # skip this part during tuning to make recrods accurate
            # this part will be pre-computed during NNVM's pre-compute optimization pass
            s[G].pragma(s[G].op.axis[0], 'debug_skip_region')
            s[kernel_pack].pragma(eps, 'debug_skip_region')
        else:
            s[G].compute_inline()
            r_a, r_b = s[kernel_pack].op.reduce_axis
            for axis in [eps, nu, r_a, r_b]:
                s[kernel_pack].unroll(axis)

            fused = s[kernel_pack].fuse(ci, co)
            bb, tt = s[kernel_pack].split(fused, 128)
            s[kernel_pack].reorder(bb, tt, eps, nu, r_a, r_b)
            s[kernel_pack].bind(bb, tvm.thread_axis("blockIdx.x"))
            s[kernel_pack].bind(tt, tvm.thread_axis("threadIdx.x"))
    else:
        kernel = kernel_pack

    if isinstance(kernel.op, tvm.tensor.ComputeOp) and "dilate" in kernel.op.tag:
        s[kernel].compute_inline()

    ##### space definition begin #####
    b1, b2, y, x = s[bgemm].op.axis
    rc = s[bgemm].op.reduce_axis[0]
    alpha = get_const_int(b1.dom.extent)

    cfg.define_split("tile_b", cfg.axis(alpha * alpha), num_outputs=4,
                     filter=lambda x: x.size[-3:] == [1, 1, 1])
    cfg.define_split("tile_y", y, num_outputs=4)
    cfg.define_split("tile_x", x, num_outputs=4)
    cfg.define_split("tile_rc", rc, num_outputs=2)
    cfg.define_knob("auto_unroll_max_step", [0, 128, 1500])
    target = tvm.target.current_target()
    if target.target_name in ['nvptx', 'rocm']:
        cfg.define_knob("unroll_explicit", [1])
    else:
        cfg.define_knob("unroll_explicit", [0, 1])
    ##### space definition end #####

    # batch gemm
    C = bgemm
    A0, B0 = kernel_pack, data_pack

    OL = s.cache_write(C, 'local')
    AA = s.cache_read(A0, 'shared', [OL])
    BB = s.cache_read(B0, 'shared', [OL])

    b = s[bgemm].fuse(b1, b2)

    # tile and bind spatial axes
    bgemm_scope, b = s[bgemm].split(b, nparts=1)
    bz, vz, tz, zi = cfg["tile_b"].apply(s, C, b)
    by, vy, ty, yi = cfg["tile_y"].apply(s, C, y)
    bx, vx, tx, xi = cfg["tile_x"].apply(s, C, x)
    s[C].bind(bz, tvm.thread_axis("blockIdx.z"))
    s[C].bind(by, tvm.thread_axis("blockIdx.y"))
    s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[C].bind(vz, tvm.thread_axis("vthread"))
    s[C].bind(vy, tvm.thread_axis("vthread"))
    s[C].bind(vx, tvm.thread_axis("vthread"))
    s[C].bind(tz, tvm.thread_axis("threadIdx.z"))
    s[C].bind(ty, tvm.thread_axis("threadIdx.y"))
    s[C].bind(tx, tvm.thread_axis("threadIdx.x"))
    s[C].reorder(bgemm_scope, bz, by, bx, vz, vy, vx, tz, ty, tx, zi, yi, xi)

    # tile reduction axes
    s[OL].compute_at(s[C], tx)
    b1, b2, y, x = s[OL].op.axis
    b = s[OL].fuse(b1, b2)
    rc, = s[OL].op.reduce_axis
    rco, rci = cfg['tile_rc'].apply(s, OL, rc)
    s[OL].reorder(rco, rci, b, y, x)

    s[AA].compute_at(s[OL], rco)
    s[BB].compute_at(s[OL], rco)

    # cooperative fetching
    for load in [AA, BB]:
        fused = s[load].fuse(*list(s[load].op.axis))
        fused, tx = s[load].split(fused, cfg["tile_x"].size[2])
        fused, ty = s[load].split(fused, cfg["tile_y"].size[2])
        fused, tz = s[load].split(fused, cfg["tile_b"].size[2])
        s[load].bind(tz, tvm.thread_axis("threadIdx.z"))
        s[load].bind(ty, tvm.thread_axis("threadIdx.y"))
        s[load].bind(tx, tvm.thread_axis("threadIdx.x"))

    s[C].pragma(bgemm_scope, 'auto_unroll_max_step', cfg['auto_unroll_max_step'].val)
    s[C].pragma(bgemm_scope, 'unroll_explicit', cfg['unroll_explicit'].val)

    # schedule inverse, output and fusion
    if output.op in s.outputs:
        OL = None
    else:
        OL = output
        s[OL].set_scope('local')
        output = s.outputs[0]

    m = alpha - 3 + 1
    n, co, h, w = s[output].op.axis
    ho, wo, hi, wi = s[output].tile(h, w, m, m)
    inverse_scope, n = s[output].split(n, nparts=1)

    fused = s[output].fuse(n, co, ho, wo)
    bb, tt = s[output].split(fused, 128)

    s[output].bind(bb, tvm.thread_axis("blockIdx.x"))
    s[output].bind(tt, tvm.thread_axis("threadIdx.x"))

    if OL is not None:
        s[OL].compute_at(s[output], tt)

    s[A].compute_inline()
    co, p, vh, vw = s[inverse].op.axis
    r_a, r_b = s[inverse].op.reduce_axis
    for axis in [vh, vw, r_a, r_b]:
        s[inverse].unroll(axis)
    s[inverse].compute_at(s[output], tt)

    return s

##### REGISTER TOPI COMPUTE / SCHEDULE FOR WINOGRAD WITH WEIGHT TRANSFORM #####
@autotvm.register_topi_compute(conv2d_winograd_without_weight_transform,
                               ['cuda', 'gpu'], ['winograd'])
def conv2d_winograd_ww(cfg, data, kernel, strides, padding, dilation, layout, out_dtype, tile_size):
    return winograd_cuda(cfg, data, kernel, strides, padding, dilation, layout, out_dtype,
                         pre_computed=True)


@autotvm.register_topi_schedule(schedule_conv2d_winograd_without_weight_transform,
                                ['cuda', 'gpu'], ['winograd'])
def schedule_conv2d_winograd_without_weight_transform_cuda(cfg, outs):
    """TOPI schedule callback"""
    s = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        if 'conv2d_nchw_winograd' in op.tag:
            schedule_winograd_cuda(cfg, s, op.output(0), pre_computed=True)

    traverse_inline(s, outs[0].op, _callback)
    return s


def winograd_NCHWc_cuda(cfg, data, kernel, strides, padding, dilation, layout, out_dtype, pre_computed):
    """Compute declaration for winograd NCHWc layout."""
    tile_size = infer_tile_size(data, kernel)
    N, CI_CHUNK, H, W, CI_BLOCK = get_const_tuple(data.shape)

    if not pre_computed: # kernel tensor is raw tensor, do strict check
        if isinstance(dilation, int):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation
        if dilation_h != 1 or dilation_w != 1:
            kernel = nn.dilate(kernel, (1, 1, dilation_h, dilation_w))

        CO, CI, KH, KW = get_const_tuple(kernel.shape)
        HPAD, WPAD, _, _ = nn.get_pad_tuple(padding, kernel)
        HSTR, WSTR = (strides, strides) if isinstance(strides, int) else strides
        assert HSTR == 1 and WSTR == 1 and HPAD == 1 and WPAD == 1 and KH == 3 and KW == 3
    else:                   # kernel tensor is pre-transfomred. this op is created by
        # alter op layout, do not check
        # dilation is not supported
        HSTR = WSTR = 1
        HPAD = WPAD = 1
        KH = KW = 3
        _, _, CI, CO = get_const_tuple(kernel.shape)

    data_pad = nn.pad(data, (0, 0, HPAD, WPAD, 0), (0, 0, HPAD, WPAD, 0), name="data_pad")

    if tile_size == 4:
        G_data = np.array([
            [1 / 4.0, 0, 0],
            [-1 / 6.0, -1 / 6.0, -1 / 6.0],
            [-1 / 6.0, 1 / 6.0, -1 / 6.0],
            [1 / 24.0, 1 / 12.0, 1 / 6.0],
            [1 / 24.0, -1 / 12.0, 1 / 6.0],
            [0, 0, 1]], dtype=np.float32)

        B_data = np.array([
            [4, 0, 0, 0, 0, 0],
            [0, -4, 4, -2, 2, 4],
            [-5, -4, -4, -1, -1, 0],
            [0, 1, -1, 2, -2, -5],
            [1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 1]], out_dtype)

        A_data = np.array([
            [1, 0, 0, 0],
            [1, 1, 1, 1],
            [1, -1, 1, -1],
            [1, 2, 4, 8],
            [1, -2, 4, -8],
            [0, 0, 0, 1]], out_dtype)
    elif tile_size == 2:
        G_data = np.array([
            [1, 0, 0],
            [1.0/2, 1.0/2, 1.0/2],
            [1.0/2, -1.0/2, 1.0/2],
            [0, 0, 1]], np.float32)

        B_data = np.array([
            [1, 0, 0, 0],
            [0, 1, -1, 1],
            [-1, 1, 1, 0],
            [0, 0, 0, -1]], out_dtype)

        A_data = np.array([
            [1, 0],
            [1, 1],
            [1, -1],
            [0, -1]], out_dtype)
    else:
        raise ValueError("Unsupported tile size for winograd: " + str(tile_size))

    m = A_data.shape[1]
    r = 3
    alpha = m + r - 1
    H = (H + 2 * HPAD - KH) // HSTR + 1
    W = (W + 2 * WPAD - KW) // WSTR + 1
    nH, nW = (H + m-1) // m, (W + m-1) // m
    P = N * nH * nW
    CO_BLOCK = cfg["tile_oc"].val
    CO_CHUNK = CO // CO_BLOCK

    # transform kernel
    if not pre_computed:
        G = const_matrix(G_data, 'G')
        r_kh = tvm.reduce_axis((0, KH), name='r_kh')
        r_kw = tvm.reduce_axis((0, KW), name='r_kw')
        kernel_pack = tvm.compute((alpha, alpha, CI, CO), lambda eps, nu, ci, co:
        tvm.sum(kernel[co][ci][r_kh][r_kw] *
                G[eps][r_kh] * G[nu][r_kw],
                axis=[r_kh, r_kw]), name='kernel_pack')
    else:
        kernel_pack = kernel

    # pack input tile
    input_tile = tvm.compute((CI, P, alpha, alpha), lambda c, p, eps, nu:
    data_pad[p // (nH * nW)][c // CI_BLOCK][p // nW % nH * m + eps][c % CI_BLOCK]
    [p % nW * m + nu], name='d')

    # transform data
    B = const_matrix(B_data)
    r_a = tvm.reduce_axis((0, alpha), 'r_a')
    r_b = tvm.reduce_axis((0, alpha), 'r_a')
    data_pack = tvm.compute((alpha, alpha, CI, P), lambda eps, nu, ci, p:
    tvm.sum(input_tile[ci][p][r_a][r_b] * B[r_a][eps] * B[r_b][nu],
            axis=[r_a, r_b]), name='data_pack')

    # do batch gemm
    ci = tvm.reduce_axis((0, CI), name='ci')
    bgemm = tvm.compute((alpha, alpha, CO, P), lambda eps, nu, co, p:
    tvm.sum(kernel_pack[eps][nu][ci][co] *
            data_pack[eps][nu][ci][p],
            axis=[ci]), name='bgemm')

    # inverse transform
    A = const_matrix(A_data)
    r_a = tvm.reduce_axis((0, alpha), 'r_a')
    r_b = tvm.reduce_axis((0, alpha), 'r_a')
    inverse = tvm.compute((CO, P, m, m), lambda co, p, vh, vw:
    tvm.sum(bgemm[r_a][r_b][co][p] * A[r_a][vh] * A[r_b][vw],
            axis=[r_a, r_b]), name='inverse')

    # output
    output = tvm.compute((N, CO_CHUNK, H, W, CO_BLOCK), lambda n, co_chunk, h, w, co_block:
    inverse[co_chunk * CO_BLOCK + co_block][n * nH * nW + (h // m) * nW + w // m][h % m][w % m],
                         name='output', tag='conv2d_NCHWc_winograd')
    cfg.add_flop(2 * N * CO * H * W * CI * KH * KW)

    return output


def schedule_winograd_NCHWc_cuda(cfg, s, output, pre_computed):
    """Schedule winograd template"""
    # get stages
    inverse = s[output].op.input_tensors[0]
    bgemm, A = s[inverse].op.input_tensors
    kernel_pack, data_pack = s[bgemm].op.input_tensors
    input_tile, B = s[data_pack].op.input_tensors
    packed_data = s[input_tile].op.input_tensors[0]

    if isinstance(packed_data.op, tvm.tensor.ComputeOp) and "pad" in packed_data.op.tag:
        pad_data = packed_data
        packed_data = pad_data.op.input_tensors[0]
    else:
        pad_data = packed_data

    if packed_data != pad_data:
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # skip this part during tuning to make recrods accurate
            # this part will be pre-computed during NNVM's pre-compute optimization pass
            s[packed_data].pragma(s[packed_data].op.axis[0], "debug_skip_region")
        else:
            if isinstance(packed_data.op, tvm.tensor.ComputeOp) and \
                    packed_data.name == 'packed_data':
                # data and kernel are not pre-computed, schedule layout transform here
                _schedule_injective(packed_data.op, s)

    # data transform
    s[B].compute_inline()

    data_l = s.cache_write(data_pack, 'local')
    eps, nu, c, p = s[data_l].op.axis
    r_a, r_b = s[data_l].op.reduce_axis
    for axis in [eps, nu, r_a, r_b]:
        s[data_l].unroll(axis)

    eps, nu, c, p = s[data_pack].op.axis
    p, pi = s[data_pack].split(p, 1)
    fused = s[data_pack].fuse(c, p)
    bb, tt = s[data_pack].split(fused, 128)
    s[data_pack].reorder(bb, tt, pi, eps, nu)
    s[data_pack].bind(bb, tvm.thread_axis("blockIdx.x"))
    s[data_pack].bind(tt, tvm.thread_axis("threadIdx.x"))

    s[data_l].compute_at(s[data_pack], pi)
    s[input_tile].compute_at(s[data_pack], pi)
    s[pad_data].compute_inline()

    # transform kernel
    if not pre_computed:
        kernel, G = s[kernel_pack].op.input_tensors
        eps, nu, ci, co = s[kernel_pack].op.axis
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # skip this part during tuning to make recrods accurate
            # this part will be pre-computed during NNVM's pre-compute optimization pass
            s[G].pragma(s[G].op.axis[0], 'debug_skip_region')
            s[kernel_pack].pragma(eps, 'debug_skip_region')
        else:
            s[G].compute_inline()
            r_a, r_b = s[kernel_pack].op.reduce_axis
            for axis in [eps, nu, r_a, r_b]:
                s[kernel_pack].unroll(axis)

            fused = s[kernel_pack].fuse(ci, co)
            bb, tt = s[kernel_pack].split(fused, 128)
            s[kernel_pack].reorder(bb, tt, eps, nu, r_a, r_b)
            s[kernel_pack].bind(bb, tvm.thread_axis("blockIdx.x"))
            s[kernel_pack].bind(tt, tvm.thread_axis("threadIdx.x"))
    else:
        kernel = kernel_pack

    if isinstance(kernel.op, tvm.tensor.ComputeOp) and "dilate" in kernel.op.tag:
        s[kernel].compute_inline()

    ##### space definition begin #####
    b1, b2, y, x = s[bgemm].op.axis
    rc = s[bgemm].op.reduce_axis[0]
    alpha = get_const_int(b1.dom.extent)

    cfg.define_split("tile_b", cfg.axis(alpha * alpha), num_outputs=4,
                     filter=lambda x: x.size[-3:] == [1, 1, 1])
    cfg.define_split("tile_y", y, num_outputs=4)
    cfg.define_split("tile_x", x, num_outputs=4)
    cfg.define_split("tile_rc", rc, num_outputs=2)
    cfg.define_knob("auto_unroll_max_step", [0, 128, 1500])
    target = tvm.target.current_target()
    if target.target_name in ['nvptx', 'rocm']:
        cfg.define_knob("unroll_explicit", [1])
    else:
        cfg.define_knob("unroll_explicit", [0, 1])
    ##### space definition end #####

    # batch gemm
    C = bgemm
    A0, B0 = kernel_pack, data_pack

    OL = s.cache_write(C, 'local')
    AA = s.cache_read(A0, 'shared', [OL])
    BB = s.cache_read(B0, 'shared', [OL])

    b = s[bgemm].fuse(b1, b2)

    # tile and bind spatial axes
    bgemm_scope, b = s[bgemm].split(b, nparts=1)
    bz, vz, tz, zi = cfg["tile_b"].apply(s, C, b)
    by, vy, ty, yi = cfg["tile_y"].apply(s, C, y)
    bx, vx, tx, xi = cfg["tile_x"].apply(s, C, x)
    s[C].bind(bz, tvm.thread_axis("blockIdx.z"))
    s[C].bind(by, tvm.thread_axis("blockIdx.y"))
    s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[C].bind(vz, tvm.thread_axis("vthread"))
    s[C].bind(vy, tvm.thread_axis("vthread"))
    s[C].bind(vx, tvm.thread_axis("vthread"))
    s[C].bind(tz, tvm.thread_axis("threadIdx.z"))
    s[C].bind(ty, tvm.thread_axis("threadIdx.y"))
    s[C].bind(tx, tvm.thread_axis("threadIdx.x"))
    s[C].reorder(bgemm_scope, bz, by, bx, vz, vy, vx, tz, ty, tx, zi, yi, xi)

    # tile reduction axes
    s[OL].compute_at(s[C], tx)
    b1, b2, y, x = s[OL].op.axis
    b = s[OL].fuse(b1, b2)
    rc, = s[OL].op.reduce_axis
    rco, rci = cfg['tile_rc'].apply(s, OL, rc)
    s[OL].reorder(rco, rci, b, y, x)

    s[AA].compute_at(s[OL], rco)
    s[BB].compute_at(s[OL], rco)

    # cooperative fetching
    for load in [AA, BB]:
        fused = s[load].fuse(*list(s[load].op.axis))
        fused, tx = s[load].split(fused, cfg["tile_x"].size[2])
        fused, ty = s[load].split(fused, cfg["tile_y"].size[2])
        fused, tz = s[load].split(fused, cfg["tile_b"].size[2])
        s[load].bind(tz, tvm.thread_axis("threadIdx.z"))
        s[load].bind(ty, tvm.thread_axis("threadIdx.y"))
        s[load].bind(tx, tvm.thread_axis("threadIdx.x"))

    s[C].pragma(bgemm_scope, 'auto_unroll_max_step', cfg['auto_unroll_max_step'].val)
    s[C].pragma(bgemm_scope, 'unroll_explicit', cfg['unroll_explicit'].val)

    # schedule inverse, output and fusion
    if output.op in s.outputs:
        OL = None
    else:
        OL = output
        s[OL].set_scope('local')
        output = s.outputs[0]

    m = alpha - 3 + 1
    n, co_chunk, h, w, co_block = s[output].op.axis
    ho, wo, hi, wi = s[output].tile(h, w, m, m)
    inverse_scope, n = s[output].split(n, nparts=1)

    fused = s[output].fuse(n, co_chunk, ho, wo)
    bb, tt = s[output].split(fused, 128 // cfg["tile_oc"].val)

    s[output].bind(bb, tvm.thread_axis("blockIdx.x"))
    s[output].bind(tt, tvm.thread_axis("threadIdx.x"))

    if OL is not None:
        s[OL].compute_at(s[output], tt)

    s[A].compute_inline()
    co, p, vh, vw = s[inverse].op.axis
    r_a, r_b = s[inverse].op.reduce_axis
    for axis in [vh, vw, r_a, r_b]:
        s[inverse].unroll(axis)
    s[inverse].compute_at(s[output], tt)

    return s

