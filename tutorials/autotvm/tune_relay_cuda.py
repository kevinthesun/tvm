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
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
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
parser.add_argument('--start', type=int, required=True)
parser.add_argument('--end', type=int, required=True)

args = parser.parse_args()
mx_model = args.model
bs = args.batch_size
start = args.start
end = args.end

def get_network(name, batch_size):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    if "resnet" in name:
        n_layer = int(name.split('-')[1])
        mod, params = relay.testing.resnet.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif "vgg" in name:
        n_layer = int(name.split('-')[1])
        mod, params = relay.testing.vgg.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif name == 'mobilenet':
        mod, params = relay.testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == 'squeezenet_v1.1':
        mod, params = relay.testing.squeezenet.get_workload(batch_size=batch_size, version='1.1', dtype=dtype)
    elif name == 'inception_v3':
        input_shape = (1, 3, 299, 299)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
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

    return mod, params, input_shape, output_shape

###########################################
# Set Tuning Options
# ------------------
# Before tuning, we apply some configurations.

#### DEVICE CONFIG ####
target = tvm.target.cuda()

#### TUNING OPTION ####
network = 'mxnet'
log_file = "conv2d_%s_%d.log" % (mx_model, bs)
tmp_log_file = "conv2d_%s_%d_%d_%d.log" % (mx_model, bs, start, end)
graph_opt_sch_file = "%s_%d_graph_opt.log" % (mx_model, bs)
dtype = 'float32'

from tvm.autotvm.record import load_from_file
wkl_set = set()
if os.path.isfile(log_file):
    for i, r in enumerate([i for i in load_from_file(log_file)]):
        wkl_set.add(r[0].task.workload)

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
        if tsk.workload in wkl_set:
            print("%s already tuned. Skipped" % str(tsk.workload))
            continue

        prefix = "[Task %d - %d %2d/%2d] " %(start, end, i+1, len(tasks))

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
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        n_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(n_trial=n_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)])

    # pick best records to a cache file
    #autotvm.record.pick_best(tmp_log_file, log_filename)
    #os.remove(tmp_log_file)

# Use graph tuner to achieve graph level optimal schedules
# Set use_DP=False if it takes too long to finish.
def tune_graph(graph, dshape, records, opt_sch_file, use_DP=True):
    target_op = [relay.nn.conv2d]
    Tuner = DPTuner if use_DP else PBQPTuner
    executor = Tuner(graph, {"data": dshape}, records, target_op, target)
    executor.benchmark_layout_transform(min_exec_num=2000)
    executor.run()
    executor.write_opt_sch2record_file(opt_sch_file)

########################################################################
# Finally, we launch tuning jobs and evaluate the end-to-end performance.

def tune_and_evaluate(tuning_opt):
    # extract workloads from relay program
    print("Extract tasks...")
    net, params, input_shape, out_shape = get_network(network, batch_size=bs)
    tasks = autotvm.task.extract_from_program(net, target=target,
                                              params=params, ops=(relay.op.nn.conv2d, relay.op.nn.conv2d_transpose))

    # run tuning tasks
    print("Tuning...")
    tasks = [tasks[i] for i in range(start, end)]
    tune_tasks(tasks, **tuning_opt)
    tune_graph(net, input_shape, log_file, graph_opt_sch_file)

    # compile kernels with history best records
    with autotvm.apply_graph_best(graph_opt_sch_file):
        print("Compile...")
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build_module.build(
                mod, target=target, params=params)

        # export library
        #tmp = tempdir()
        #filename = "net.tar"
        #lib.export_library(tmp.relpath(filename))

        # load parameters
        ctx = tvm.gpu(0)#context(str(target), 0)
        module = runtime.create(graph, lib, ctx)
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input('data', data_tvm)
        module.set_input(**params)

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=1000, repeat=1)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
              (np.mean(prof_res), np.std(prof_res)))

# We do not run the tuning in our webpage server since it takes too long.
# Uncomment the following line to run it by yourself.

tune_and_evaluate(tuning_option)

######################################################################
# Sample Output
# -------------
# The tuning needs to compile many programs and extract feature from them.
# So a high performance CPU is recommended. One sample output is listed below.
# It takes about 4 hours to get the following output on a 32T AMD Ryzen Threadripper.
# The tuning target is NVIDIA 1080 Ti.
# (You can see some errors during compilation. If the tuning is not stuck, it is okay.)
#
# .. code-block:: bash
#
#    Extract tasks...
#    Tuning...
#    [Task  1/12]  Current/Best:  541.83/3570.66 GFLOPS | Progress: (960/2000) | 1001.31 s Done.
#    [Task  2/12]  Current/Best:    0.56/ 803.33 GFLOPS | Progress: (704/2000) | 608.08 s Done.
#    [Task  3/12]  Current/Best:  103.69/1141.25 GFLOPS | Progress: (768/2000) | 702.13 s Done.
#    [Task  4/12]  Current/Best: 2905.03/3925.15 GFLOPS | Progress: (864/2000) | 745.94 sterminate called without an active exception
#    [Task  4/12]  Current/Best: 2789.36/3925.15 GFLOPS | Progress: (1056/2000) | 929.40 s Done.
#    [Task  5/12]  Current/Best:   89.06/1076.24 GFLOPS | Progress: (704/2000) | 601.73 s Done.
#    [Task  6/12]  Current/Best:   40.39/2129.02 GFLOPS | Progress: (1088/2000) | 1125.76 s Done.
#    [Task  7/12]  Current/Best: 4090.53/5007.02 GFLOPS | Progress: (800/2000) | 903.90 s Done.
#    [Task  8/12]  Current/Best:    4.78/1272.28 GFLOPS | Progress: (768/2000) | 749.14 s Done.
#    [Task  9/12]  Current/Best: 1391.45/2325.08 GFLOPS | Progress: (992/2000) | 1084.87 s Done.
#    [Task 10/12]  Current/Best: 1995.44/2383.59 GFLOPS | Progress: (864/2000) | 862.60 s Done.
#    [Task 11/12]  Current/Best: 4093.94/4899.80 GFLOPS | Progress: (224/2000) | 240.92 sterminate called without an active exception
#    [Task 11/12]  Current/Best: 3487.98/4909.91 GFLOPS | Progress: (480/2000) | 534.96 sterminate called without an active exception
#    [Task 11/12]  Current/Best: 4636.84/4912.17 GFLOPS | Progress: (1184/2000) | 1381.16 sterminate called without an active exception
#    [Task 11/12]  Current/Best:   50.12/4912.17 GFLOPS | Progress: (1344/2000) | 1602.81 s Done.
#    [Task 12/12]  Current/Best: 3581.31/4286.30 GFLOPS | Progress: (736/2000) | 943.52 s Done.
#    Compile...
#    Evaluate inference time cost...
#    Mean inference time (std dev): 1.07 ms (0.05 ms)
#
# As a reference baseline, the time cost of MXNet + TensorRT on resnet-18 is 1.30ms. So we are a little faster.

######################################################################
#
# .. note:: **Experiencing Difficulties?**
#
#   The auto tuning module is error-prone. If you always see " 0.00/ 0.00 GFLOPS",
#   then there must be something wrong.
#
#   First, make sure you set the correct configuration of your device.
#   Then, you can print debug information by adding these lines in the beginning
#   of the script. It will print every measurement result, where you can find useful
#   error messages.
#
#   .. code-block:: python
#
#      import logging
#      logging.getLogger('autotvm').setLevel(logging.DEBUG)
#
#   Finally, always feel free to ask our community for help on https://discuss.tvm.ai


#################################################################
# Scale up measurement by using multiple devices
# ----------------------------------------------
#
# If you have multiple devices, you can use all of them for measurement.
# TVM uses the RPC Tracker to manage distributed devices.
# The RPC Tracker is a centralized master node. We can register all devices to
# the tracker. For example, if we have 10 GPU cards, we can register all of them
# to the tracker, and run 10 measurements in parallel, accelerating the tuning process.
#
# To start an RPC tracker, run this command on the host machine. The tracker is
# required during the whole tuning process, so we need to open a new terminal for
# this command:
#
# .. code-block:: bash
#
#   python -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190
#
# The expected output is
#
# .. code-block:: bash
#
#   INFO:RPCTracker:bind to 0.0.0.0:9190
#
# Then open another new terminal for the RPC server. We need to start one server
# for each dedicated device. We use a string key to distinguish the types of devices.
# You can pick a name you like.
# (Note: For rocm backend, there are some internal errors with the compiler,
# we need to add `--no-fork` to the argument list.)
#
# .. code-block:: bash
#
#     python -m tvm.exec.rpc_server --tracker=0.0.0.0:9190 --key=1080ti
#
# After registering devices, we can confirm it by querying rpc_tracker
#
# .. code-block:: bash
#
#   python -m tvm.exec.query_rpc_tracker --host=0.0.0.0 --port=9190
#
# For example, if we have four 1080ti, two titanx and one gfx900, the output can be
#
# .. code-block:: bash
#
#    Queue Status
#    ----------------------------------
#    key          total  free  pending
#    ----------------------------------
#    1080ti       4      4     0
#    titanx       2      2     0
#    gfx900       1      1     0
#    ----------------------------------
#
# Finally, we need to change the tuning option to use RPCRunner. Use the code below
# to replace the corresponding part above.

