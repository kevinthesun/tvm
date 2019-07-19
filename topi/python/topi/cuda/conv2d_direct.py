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
"""The templates for cuda conv2d operator"""
import tvm
from tvm import autotvm
from .injective import _schedule_injective
from ..util import get_const_tuple

def schedule_direct_cuda(cfg, s, conv):
    """schedule optimized for batch size = 1"""
    ##### space definition begin #####
    n, f, y, x = s[conv].op.axis
    rc, ry, rx = s[conv].op.reduce_axis
    cfg.define_split("tile_f", f, num_outputs=4)
    cfg.define_split("tile_y", y, num_outputs=4)
    cfg.define_split("tile_x", x, num_outputs=4)
    cfg.define_split("tile_rc", rc, num_outputs=2)
    cfg.define_split("tile_ry", ry, num_outputs=2)
    cfg.define_split("tile_rx", rx, num_outputs=2)
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])

    target = tvm.target.current_target()
    if target.target_name in ['nvptx', 'rocm']:
        cfg.define_knob("unroll_explicit", [1])
    else:
        cfg.define_knob("unroll_explicit", [0, 1])

    # fallback support
    if cfg.is_fallback:
        ref_log = autotvm.tophub.load_reference_log(
            target.target_name, target.model, 'conv2d', 'direct')
        cfg.fallback_with_reference_log(ref_log)
    ##### space definition end #####

    pad_data, kernel = s[conv].op.input_tensors

    s[pad_data].compute_inline()
    if isinstance(kernel.op, tvm.tensor.ComputeOp) and 'dilate' in kernel.op.tag:
        s[kernel].compute_inline()

    if conv.op in s.outputs:
        output = conv
        OL = s.cache_write(conv, 'local')
    else:
        output = s.outputs[0].output(0)
        s[conv].set_scope('local')
        OL = conv

    # create cache stage
    AA = s.cache_read(pad_data, 'shared', [OL])
    WW = s.cache_read(kernel, 'shared', [OL])

    # tile and bind spatial axes
    n, f, y, x = s[output].op.axis
    kernel_scope, n = s[output].split(n, nparts=1)

    bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)
    by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
    bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)

    bf = s[output].fuse(n, bf)
    s[output].bind(bf, tvm.thread_axis("blockIdx.z"))
    s[output].bind(by, tvm.thread_axis("blockIdx.y"))
    s[output].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[output].bind(vf, tvm.thread_axis("vthread"))
    s[output].bind(vy, tvm.thread_axis("vthread"))
    s[output].bind(vx, tvm.thread_axis("vthread"))
    s[output].bind(tf, tvm.thread_axis("threadIdx.z"))
    s[output].bind(ty, tvm.thread_axis("threadIdx.y"))
    s[output].bind(tx, tvm.thread_axis("threadIdx.x"))
    s[output].reorder(bf, by, bx, vf, vy, vx, tf, ty, tx, fi, yi, xi)
    s[OL].compute_at(s[output], tx)

    # tile reduction axes
    n, f, y, x = s[OL].op.axis
    rc, ry, rx = s[OL].op.reduce_axis
    rco, rci = cfg['tile_rc'].apply(s, OL, rc)
    ryo, ryi = cfg['tile_ry'].apply(s, OL, ry)
    rxo, rxi = cfg['tile_rx'].apply(s, OL, rx)
    s[OL].reorder(rco, ryo, rxo, rci, ryi, rxi, n, f, y, x)

    s[AA].compute_at(s[OL], rxo)
    s[WW].compute_at(s[OL], rxo)

    # cooperative fetching
    for load in [AA, WW]:
        n, f, y, x = s[load].op.axis
        fused = s[load].fuse(n, f, y, x)
        tz, fused = s[load].split(fused, nparts=cfg["tile_f"].size[2])
        ty, fused = s[load].split(fused, nparts=cfg["tile_y"].size[2])
        tx, fused = s[load].split(fused, nparts=cfg["tile_x"].size[2])
        s[load].bind(tz, tvm.thread_axis("threadIdx.z"))
        s[load].bind(ty, tvm.thread_axis("threadIdx.y"))
        s[load].bind(tx, tvm.thread_axis("threadIdx.x"))

    # unroll
    s[output].pragma(kernel_scope, 'auto_unroll_max_step', cfg['auto_unroll_max_step'].val)
    s[output].pragma(kernel_scope, 'unroll_explicit', cfg['unroll_explicit'].val)

    N, CO, OH, OW = get_const_tuple(output.shape)
    _, KH, KW, CI = get_const_tuple(kernel.shape)
    cfg.add_flop(2 * N * OH * OW * CO * CI * KH * KW)


def schedule_conv2d_nchwc_cuda(cfg, s, conv):
    """Schedule conv2d NCHWc template"""
    # tile and bind spatial axes
    n, f, y, x, c = s[conv].op.axis
    # This is a hack.
    #cfg.define_split("tile_n", cfg.axis(n), num_outputs=4)
    cfg.define_split("tile_f", cfg.axis(f), num_outputs=4)
    cfg.define_split("tile_y", cfg.axis(y), num_outputs=4)
    cfg.define_split("tile_x", cfg.axis(x), num_outputs=4)

    rc, ry, rx, rc_block = s[conv].op.reduce_axis
    cfg.define_split("tile_rc", cfg.axis(rc), num_outputs=2)
    cfg.define_split("tile_ry", cfg.axis(ry), num_outputs=2)
    cfg.define_split("tile_rx", cfg.axis(rx), num_outputs=2)

    packed_data, packed_kernel = conv.op.input_tensors

    if isinstance(packed_data.op, tvm.tensor.ComputeOp) and "pad" in packed_data.op.tag:
        pad_data = packed_data
        packed_data = pad_data.op.input_tensors[0]
    else:
        pad_data = packed_data

    if autotvm.GLOBAL_SCOPE.in_tuning:
        # skip this part during tuning to make recrods accurate
        # this part will be pre-computed during NNVM's pre-compute optimization pass
        s[packed_data].pragma(s[packed_data].op.axis[0], "debug_skip_region")
        s[packed_kernel].pragma(s[packed_kernel].op.axis[0], "debug_skip_region")
    else:
        if isinstance(packed_kernel.op, tvm.tensor.ComputeOp) and \
                packed_kernel.name == 'packed_kernel':
            # data and kernel are not pre-computed, schedule layout transform here
            _schedule_injective(packed_data.op, s)
            _schedule_injective(packed_kernel.op, s)

    if pad_data != packed_data:
        s[pad_data].compute_inline()

    if conv.op in s.outputs:
        output = conv
        OL = s.cache_write(conv, 'local')
    else:
        output = s.outputs[0].output(0)
        s[conv].set_scope('local')
        OL = conv

    # create cache stage
    AA = s.cache_read(pad_data, 'shared', [OL])
    WW = s.cache_read(packed_kernel, 'shared', [OL])


    # tile and bind spatial axes
    n, f, y, x, c = s[output].op.axis
    # this is the scope to attach global config inside this kernel
    kernel_scope, n = s[output].split(n, nparts=1)

    bn, vn, tn, ni = cfg["tile_n"].apply(s, output, n)
    bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)
    by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
    bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)

    s[output].reorder(bn, bf, by, bx, vn, vf, vy, vx, tn, tf, ty, tx, ni, fi, yi, xi)
    s[output].bind(bn, tvm.thread_axis("blockIdx.z"))
    s[output].bind(bf, tvm.thread_axis("blockIdx.y"))
    s[output].bind(s[output].fuse(by, bx), tvm.thread_axis("blockIdx.x"))
    s[output].bind(vn, tvm.thread_axis("vthread"))
    s[output].bind(vf, tvm.thread_axis("vthread"))
    s[output].bind(vy, tvm.thread_axis("vthread"))
    s[output].bind(vx, tvm.thread_axis("vthread"))

    cfg.define_knob("fuse_yx", [0, 1]) # fuse ty,tx or tn,tf
    if cfg["fuse_yx"].val:
        s[output].bind(tn, tvm.thread_axis("threadIdx.z"))
        s[output].bind(tf, tvm.thread_axis("threadIdx.y"))
        tyx = s[output].fuse(ty, tx)
        s[output].bind(tyx, tvm.thread_axis("threadIdx.x"))
        s[OL].compute_at(s[output], tyx)

        # number of threads
        n_tz = cfg["tile_n"].size[2]
        n_ty = cfg["tile_f"].size[2]
        n_tx = cfg["tile_y"].size[2] * cfg["tile_x"].size[2]
    else:
        s[output].bind(s[output].fuse(tn, tf), tvm.thread_axis("threadIdx.z"))
        s[output].bind(ty, tvm.thread_axis("threadIdx.y"))
        s[output].bind(tx, tvm.thread_axis("threadIdx.x"))
        s[OL].compute_at(s[output], tx)
        # number of threads
        n_tz = cfg["tile_n"].size[2] * cfg["tile_f"].size[2]
        n_ty = cfg["tile_y"].size[2]
        n_tx = cfg["tile_x"].size[2]

    # tile and bind reduction axes
    n, f, y, x, c = s[OL].op.axis
    rc, ry, rx, rc_block = s[OL].op.reduce_axis
    rco, rci = cfg['tile_rc'].apply(s, OL, rc)
    ryo, ryi = cfg['tile_ry'].apply(s, OL, ry)
    rxo, rxi = cfg['tile_rx'].apply(s, OL, rx)

    s[OL].reorder(rco, ryo, rxo, rci, ryi, rxi, n, f, y, x, c, rc_block)

    cfg.define_reorder("reorder_inner", [rco, ryo, rxo], policy="all")
    cfg["reorder_inner"].apply(s, OL, [rco, ryo, rxo])
    cfg["reorder_inner"].apply(s, OL, [rci, ryi, rxi])

    cache_loc = [rco, ryo, rxo][cfg["reorder_inner"].perm[-1]]
    s[AA].compute_at(s[OL], cache_loc)
    s[WW].compute_at(s[OL], cache_loc)

    # cooperative fetching
    for load in [AA, WW]:
        tile_ic = cfg["tile_ic"].val
        upper = 16
        factor = upper
        for i in range(tile_ic, 0, -1):
            if i <= upper and tile_ic % i == 0:
                factor = i
                break
        c = s[load].op.axis[-1]
        c_outer, c = s[load].split(c, factor=factor)
        s[load].vectorize(c)
        fused = s[load].op.axis[:-1] + [c_outer]
        fused = s[load].fuse(*fused)

        fused, tx = s[load].split(fused, factor=n_tx)
        fused, ty = s[load].split(fused, factor=n_ty)
        fused, tz = s[load].split(fused, factor=n_tz)
        s[load].bind(tz, tvm.thread_axis("threadIdx.z"))
        s[load].bind(ty, tvm.thread_axis("threadIdx.y"))
        s[load].bind(tx, tvm.thread_axis("threadIdx.x"))

    # double buffer
    cfg.define_knob('AA_double_buffer', [0, 1])
    cfg.define_knob('WW_double_buffer', [0, 1])
    if cfg['AA_double_buffer'].val:
        s[AA].double_buffer()
    if cfg['WW_double_buffer'].val:
        s[WW].double_buffer()

    # unroll
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])
    s[output].pragma(kernel_scope, 'auto_unroll_max_step',
                     cfg['auto_unroll_max_step'].val)
    s[output].pragma(kernel_scope, 'unroll_explicit', False)

    N, CO_C, OH, OW, CO_B = get_const_tuple(packed_data.shape)
    _, CI_C, KH, KW, _, CI_B = get_const_tuple(packed_kernel.shape)
    CO = CO_C * CO_B
    CI = CI_C * CI_B
    cfg.add_flop(2 * N * OH * OW * CO * CI * KH * KW)

    return s
