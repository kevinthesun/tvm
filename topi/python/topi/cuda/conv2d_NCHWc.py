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
"""The templates for cuda conv2d_NCHWc operator"""
import tvm
from tvm import autotvm
from .injective import _schedule_injective


def schedule_conv2d_nchwc_cuda(cfg, s, conv):
    """Schedule conv2d NCHWc template"""
    # tile and bind spatial axes
    n, f, y, x, c = s[conv].op.axis
    cfg.define_split("tile_n", cfg.axis(n), num_outputs=4)
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
        OL = s.cache_write(conv, 'local')
    else:
        OL = conv

    # create cache stage
    AA = s.cache_read(pad_data, 'shared', [OL])
    WW = s.cache_read(packed_kernel, 'shared', [OL])

    #s[conv].set_scope('local')

    # tile and bind spatial axes
    n, f, y, x, c = s[conv].op.axis
    # this is the scope to attach global config inside this kernel
    kernel_scope, n = s[conv].split(n, nparts=1)

    bn, vn, tn, ni = cfg["tile_n"].apply(s, conv, n)
    bf, vf, tf, fi = cfg["tile_f"].apply(s, conv, f)
    by, vy, ty, yi = cfg["tile_y"].apply(s, conv, y)
    bx, vx, tx, xi = cfg["tile_x"].apply(s, conv, x)

    s[conv].reorder(bn, bf, by, bx, vn, vf, vy, vx, tn, tf, ty, tx, ni, fi, yi, xi)
    s[conv].bind(bn, tvm.thread_axis("blockIdx.z"))
    s[conv].bind(bf, tvm.thread_axis("blockIdx.y"))
    s[conv].bind(s[conv].fuse(by, bx), tvm.thread_axis("blockIdx.x"))
    s[conv].bind(vn, tvm.thread_axis("vthread"))
    s[conv].bind(vf, tvm.thread_axis("vthread"))
    s[conv].bind(vy, tvm.thread_axis("vthread"))
    s[conv].bind(vx, tvm.thread_axis("vthread"))

    cfg.define_knob("fuse_yx", [0, 1]) # fuse ty,tx or tn,tf
    if cfg["fuse_yx"].val:
        s[conv].bind(tn, tvm.thread_axis("threadIdx.z"))
        s[conv].bind(tf, tvm.thread_axis("threadIdx.y"))
        tyx = s[conv].fuse(ty, tx)
        s[conv].bind(tyx, tvm.thread_axis("threadIdx.x"))
        s[OL].compute_at(s[conv], tyx)

        # number of threads
        n_tz = cfg["tile_n"].size[2]
        n_ty = cfg["tile_f"].size[2]
        n_tx = cfg["tile_y"].size[2] * cfg["tile_x"].size[2]
    else:
        s[conv].bind(s[conv].fuse(tn, tf), tvm.thread_axis("threadIdx.z"))
        s[conv].bind(ty, tvm.thread_axis("threadIdx.y"))
        s[conv].bind(tx, tvm.thread_axis("threadIdx.x"))
        s[OL].compute_at(s[conv], tx)
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
        c = s[load].op.axis[-1]
        c_outer, c = s[load].split(c, factor=4)
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
    s[conv].pragma(kernel_scope, 'auto_unroll_max_step',
                   cfg['auto_unroll_max_step'].val)
    s[conv].pragma(kernel_scope, 'unroll_explicit', False)

    return s