"""
Auto-tuning a convolutional network for NVIDIA GPU
==================================================
**Author**: `Lianmin Zheng <https://github.com/merrymercy>`_, `Eddie Yan <https://github.com/eqy/>`_

Auto-tuning for specific devices and workloads is critical for getting the
best performance. This is a tutorial on how to tune a whole convolutional
network for NVIDIA GPU.

The operator implementation for NVIDIA GPU in TVM is written in template form.
The template has many tunable knobs (tile factor, unrolling, etc).
We will tune all convolution and depthwise convolution operators
in the neural network. After tuning, we produce a log file which stores
the best knob values for all required operators. When the tvm compiler compiles
these operators, it will query this log file to get the best knob values.

We also released pre-tuned parameters for some NVIDIA GPUs. You can go to
`NVIDIA GPU Benchmark <https://github.com/dmlc/tvm/wiki/Benchmark#nvidia-gpu>`_
to see the results.
"""

######################################################################
# Install dependencies
# --------------------
# To use the autotvm package in tvm, we need to install some extra dependencies.
# (change "3" to "2" if you use python2):
#
# .. code-block:: bash
#
#   pip3 install --user psutil xgboost tornado
#
# To make tvm run faster during tuning, it is recommended to use cython
# as FFI of tvm. In the root directory of tvm, execute:
#
# .. code-block:: bash
#
#   pip3 install --user cython
#   sudo make cython3
#
# Now return to python code. Import packages.

import os
import multiprocessing

import numpy as np

import tvm
import argparse
from tvm import autotvm
from tvm import relay
from tvm.autotvm.record import load_from_file
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
import tvm.contrib.graph_runtime as runtime

#################################################################
# Define Network
# --------------
# First we need to define the network in relay frontend API.
# We can load some pre-defined network from :code:`nnvm.testing`.
# We can also load models from MXNet, ONNX and TensorFlow.

parser = argparse.ArgumentParser(description='Search convolution workload.')
parser.add_argument('--model', type=str, required=True,
                    help="Pretrained model from gluon model zoo.")
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--num_workers', type=int, required=True)

args = parser.parse_args()
mx_model = args.model
bs = args.batch_size
num_workers = args.num_workers

def get_network(name, batch_size):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    if "resnet" in name:
        n_layer = int(name.split('-')[1])
        net, params = relay.testing.resnet.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif "vgg" in name:
        n_layer = int(name.split('-')[1])
        net, params = relay.testing.vgg.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif name == 'mobilenet':
        net, params = relay.testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == 'squeezenet_v1.1':
        net, params = relay.testing.squeezenet.get_workload(batch_size=batch_size, version='1.1', dtype=dtype)
    elif name == 'inception_v3':
        input_shape = (1, 3, 299, 299)
        net, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == 'mxnet':
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model
        if "inceptionv3" in mx_model:
            input_shape = (batch_size, 3, 299, 299)
        block = get_model(mx_model, pretrained=True)
        net, params = relay.frontend.from_mxnet(block, shape={'data': input_shape}, dtype=dtype)
        net = relay.Function(net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs)
    elif name == "gluoncv":
        from gluoncv import model_zoo
        input_shape = (batch_size, 3, 512, 512)
        block = model_zoo.get_model(mx_model, pretrained=True)
        net, params = relay.frontend.from_mxnet(block, shape={'data': input_shape}, dtype=dtype)
    elif name == 'conv2d':
        batch_size = 16
        in_channel = 256
        out_channel = 512
        in_height, in_width = 14, 14
        data_shape = (batch_size, in_channel, in_height, in_width)
        kernel_size = (3, 3)
        kernel_shape = (out_channel, in_channel,) + kernel_size
        padding = (1, 1)
        strides = (2, 2)
        dilation = (1, 1)

        data = relay.var("data", shape=data_shape)
        weight = relay.var("weight", shape=kernel_shape)
        out = relay.nn.conv2d(data, weight=weight, channels=out_channel, kernel_size=kernel_size, strides=strides, padding=padding, dilation=dilation)
        out = relay.Function(relay.ir_pass.free_vars(out), out)
        net, params = relay.testing.create_workload(out)
        input_shape = data_shape
        output_shape = (batch_size, out_channel, 7, 7)
    else:
        raise ValueError("Unsupported network: " + name)

    return net, params, input_shape, output_shape

###########################################
# Set Tuning Options
# ------------------
# Before tuning, we apply some configurations.

#### DEVICE CONFIG ####
target = tvm.target.cuda()

#### TUNING OPTION ####
network = 'mxnet'
log_file = "conv2d_%s_%d.log" % (mx_model, bs)
dtype = 'float32'

tuning_option = {
    'log_filename': log_file,

    'tuner': 'xgb',
    'n_trial': 5000,
    'early_stopping': 4000,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=10, n_parallel=2),
        runner=autotvm.LocalRunner(number=10, repeat=1, timeout=20, min_repeat_ms=2000),
        #runner=autotvm.RPCRunner(
        #    '1080ti',  # change the device key to your key
        #    '0.0.0.0', 9190,
        #    number=20, repeat=3, timeout=4, min_repeat_ms=150)
    ),
    'try_winograd': True,
    'use_transfer_learning': True
}

####################################################################
#
# .. note:: How to set tuning options
#
#   In general, the default value provided here works well.
#
#   If you have large time budget, you can set :code:`n_trial`, :code:`early_stopping` larger,
#   which makes the tuning runs longer.
#
#   If you have multiple devices, you can use all of them for measurement to
#   accelerate the tuning process. (see the 'Scale up measurement` section below).
#

###################################################################
# Begin Tuning
# ------------
# Now we can extract tuning tasks from the network and begin tuning.
# Here, we provide a simple utility function to tune a list of tasks.
# This function is just an initial implementation which tunes them in sequential order.
# We will introduce a more sophisticated tuning scheduler in the future.

# You can skip the implementation of this function for this tutorial.
def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=2000,
               early_stopping=None,
               log_filename='tuning.log',
               use_transfer_learning=False,
               try_winograd=True):
    for i in range(len(tasks)):
        tsk = autotvm.task.create(tasks[i].name, tasks[i].args,
                                  tasks[i].target, tasks[i].target_host, 'NCHWc')
        tasks[i] = tsk

    if try_winograd:
        for i in range(len(tasks)):
            try:  # try winograd template
                tsk = autotvm.task.create(tasks[i].name, tasks[i].args,
                                          tasks[i].target, tasks[i].target_host, 'winograd_NCHWc')
                input_channel = tsk.workload[1][1]
                if input_channel >= 64:
                    tasks[i] = tsk
            except Exception:
                pass

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " %(i+1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(log_file))

        # do tuning
        n_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(n_trial=n_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(log_file)])

    # pick best records to a cache file
    #autotvm.record.pick_best(tmp_log_file, log_filename)
    #os.remove(tmp_log_file)


########################################################################
# Finally, we launch tuning jobs and evaluate the end-to-end performance.

def tune_and_evaluate(tuning_opt):
    # extract workloads from relay program
    print("Extract tasks...")
    net, params, input_shape, out_shape = get_network(network, batch_size=bs)
    tasks = autotvm.task.extract_from_program(net, target=target,
                                              params=params, ops=(relay.op.nn.conv2d, relay.op.nn.conv2d_transpose))

    # run tuning tasks
    end_idx = 0
    tsk_slice = len(tasks) // num_workers
    remain = len(tasks) % num_workers
    final_cmd= ""
    for i in range(num_workers):
        start_idx = end_idx
        end_idx = start_idx + tsk_slice
        if remain > 0:
            end_idx += 1
            remain -= 1
        end_idx = min(end_idx, len(tasks))
        if end_idx > start_idx:
            cmd = "CUDA_VISIBLE_DEVICES=%s python3 tune_relay_cuda.py --model %s --batch_size %d --start %d --end %d" \
                  % (i, mx_model, bs, start_idx, end_idx)
            final_cmd += cmd + " & "

    os.system(final_cmd[:-3])

tune_and_evaluate(tuning_option)

