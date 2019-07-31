import numpy as np
import tvm
import time
import topi

from tvm.contrib import graph_runtime
from tvm.autotvm.record import load_from_file

run_times = 1000

log_file = "single_sch"
for i, r in enumerate([i for i in load_from_file(log_file)]):
    cfg = r[0].config

n = tvm.var("n")
ic = 3
oc = 64
ih, iw = 224, 224
kh, kw = (7, 7)
padding = (3, 3)
strides = (2, 2)
dilation = (1, 1)

ic_bn = cfg["tile_ic"].val
oc_bn = cfg["tile_oc"].val
oh = (ih - kh + padding[0] + padding[0]) // strides[0] + 1
ow = (iw - kw + padding[1] + padding[1]) // strides[1] + 1

target = "cuda"
img_shape = (ic // ic_bn, ih, iw, ic_bn)
dshape = (n,) + img_shape
kshape = (oc // oc_bn, ic // ic_bn, kh, kw, oc_bn, ic_bn)
data_layout = "NCHW%dc" % ic_bn
out_layout = "NCHW%dc" % oc_bn
ctx = tvm.gpu()

data = tvm.placeholder(dshape, name="data")
kernel = tvm.placeholder(kshape, name="weight")
out = topi.cuda.conv2d.conv2d_NCHWc_cuda(cfg, data, kernel, strides, padding, dilation, data_layout, out_layout, "float32")
s = topi.cuda.conv2d.schedule_conv2d_NCHWc_cuda(cfg, [out])

f = tvm.build(s, [data, kernel, out], target=target)

bs_list = [1, 2, 3, 16, 64]
for actual_bs in bs_list:
    actual_dshape = (actual_bs,) + img_shape
    actual_oshape = (actual_bs,) + (oc // oc_bn, oh, ow, oc_bn)
    d = tvm.nd.array(np.random.uniform(size=actual_dshape).astype("float32"), ctx)
    k = tvm.nd.array(np.random.uniform(size=kshape).astype("float32"), ctx)
    o = tvm.nd.empty(actual_oshape, "float32", ctx)

    time_f = f.time_evaluator(f.entry_name, ctx, number=1000)
    cost = time_f(d, k, o).mean * 1000
    print("bs %d: %f" % (actual_bs, cost / actual_bs))
