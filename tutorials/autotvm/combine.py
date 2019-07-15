import os
import argparse

parser = argparse.ArgumentParser(description='Search convolution workload.')
parser.add_argument('--model', type=str, required=True,
                    help="Pretrained model from gluon model zoo.")
parser.add_argument('--batch_size', type=int, required=True)

args = parser.parse_args()
mx_model = args.model
bs = args.batch_size

prefix = "conv2d_%s_%d" % (mx_model, bs)
log_file = "%s.log" % prefix

for f in os.listdir():
    if f.startswoth(prefix + "_"):
        with open(log_file, 'a') as out_f:
            with open(f, 'r') as in_f:
                for line in in_f:
                    out_f.write(line)
