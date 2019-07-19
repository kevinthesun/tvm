import numpy as np
import nnvm
import tvm
import time

import tvm.relay.testing
from nnvm import symbol as sym
from nnvm.testing import utils
from tvm.contrib import graph_runtime
from tvm.contrib.debugger import debug_runtime
from tvm import relay, autotvm

run_times = 1000

batch_size = relay.Any()
in_channel = 3
out_channel = 64
in_height = in_width = 224
kernel_size = (7, 7)
padding = (3, 3)
strides = (2, 2)
dilation = (1, 1)

target = "cuda"
image_shape = (in_channel, in_height, in_width)
data_shape = (batch_size,) + image_shape
weight_shape = (out_channel, in_channel,) + kernel_size


data = relay.var("data", shape=data_shape)
weight = relay.var("weight", shape=weight_shape)
bias = relay.var("bias", shape=(out_channel,))
out = relay.nn.conv2d(data, weight, channels=out_channel, strides=strides, kernel_size=kernel_size,
                      padding=padding, dilation=dilation)
#out = relay.nn.bias_add(out, bias, axis=1)
net = relay.Function(relay.analysis.free_vars(out), out)

params = {"weight": tvm.nd.array(np.random.uniform(0, 255, size=weight_shape).astype("float32"), ctx=tvm.cpu())}


ctx = tvm.gpu()
opt_level = 3

#autotvm.GLOBAL_SCOPE.in_tuning = True
tvm.autotvm.task.DispatchContext.current = tvm.autotvm.apply_history_best("single_sch")
with relay.build_config(opt_level=opt_level):
    graph, lib, params = relay.build_module.build(net, target=target,  params=params)

print(graph)
#with nnvm.compiler.build_config(opt_level=opt_level):
#    graph, lib, params = nnvm.compiler.build(net, target=target,  params=params, shape={"data": data_shape})

"""
module = graph_runtime.create(graph, lib, ctx)
data_array = np.random.uniform(0, 255, size=data_shape).astype("float32")
input_data = tvm.nd.array(data_array, ctx=ctx)
module.set_input('data', input_data)
module.set_input(**params)

# Warmup
for _ in range(10):
    module.run()

ftimer = module.module.time_evaluator("run", ctx, number=run_times, repeat=1)
prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
print(np.mean(prof_res))
"""
