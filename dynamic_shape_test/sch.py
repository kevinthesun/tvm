import tvm

n = tvm.var("n")
a = tvm.placeholder((10,), name="a")
b = tvm.compute(a.shape, lambda i: a[i] + 1)

s = tvm.create_schedule(b.op)
s[b].split(s[b].op.axis[0], 3)

tvm.lower(s, [a, b])
