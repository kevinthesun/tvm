import numpy as np
import tvm
import time

import tvm.relay.testing
from nnvm import symbol as sym
from nnvm.testing import utils
from tvm.contrib import graph_runtime
from tvm import relay, autotvm


batch_size = tvm.var("n")
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

log_file = "conv2d.log"

tuning_option = {
    'log_filename': log_file,

    'tuner': 'xgb',
    'n_trial': 2000,
    'early_stopping': 1000,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=10),
        runner=autotvm.LocalRunner(number=10, repeat=1, timeout=20, min_repeat_ms=2000),
        #runner=autotvm.RPCRunner(
        #    '1080ti',  # change the device key to your key
        #    '0.0.0.0', 9190,
        #    number=20, repeat=3, timeout=4, min_repeat_ms=150)
    ),
    'try_winograd': True,
    'use_transfer_learning': True
}
