"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import numpy as np
import paddle

import os
import argparse
import paddle
import random
paddle.seed(0)
np.random.seed(0)
random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--static', default=False, action="store_true")
parser.add_argument(
    '--prim', default=False, action="store_true")
args = parser.parse_args()

if args.static is True:
    print("============= 静态图静态图静态图静态图静态图 =============")
    paddle.enable_static()
    if args.prim:
        paddle.incubate.autograd.enable_prim()
        print("============= prim prim prim prim prim  =============")
else:
    print("============= 动态图动态图动态图动态图动态图 =============")


task_name = os.path.basename(__file__).split(".")[0]
# 创建任务日志文件夹
log_dir = f"./params/{task_name}"
os.makedirs(f"{log_dir}", exist_ok=True)
# General parameters
n = 2
precision_train = 10
precision_test = 30
hard_constraint = True
weights = 100  # if hard_constraint == False
iterations = 5000
parameters = [1e-3, 3, 150, "sin"]

# Define sine function
if dde.backend.backend_name == "pytorch":
    sin = dde.backend.pytorch.sin
elif dde.backend.backend_name == "paddle":
    sin = paddle.sin
else:
    from deepxde.backend import tf
    sin = tf.sin

learning_rate, num_dense_layers, num_dense_nodes, activation = parameters


def pde(x, y):
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)

    f = k0 ** 2 * sin(k0 * x[:, 0:1]) * sin(k0 * x[:, 1:2])
    return -dy_xx - dy_yy - k0 ** 2 * y - f


def func(x):
    return np.sin(k0 * x[:, 0:1]) * np.sin(k0 * x[:, 1:2])


def transform(x, y):
    res = x[:, 0:1] * (1 - x[:, 0:1]) * x[:, 1:2] * (1 - x[:, 1:2])
    return res * y


def boundary(_, on_boundary):
    return on_boundary


geom = dde.geometry.Rectangle([0, 0], [1, 1])
k0 = 2 * np.pi * n
wave_len = 1 / n

hx_train = wave_len / precision_train
nx_train = int(1 / hx_train)

hx_test = wave_len / precision_test
nx_test = int(1 / hx_test)

if hard_constraint == True:
    bc = []
else:
    bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary)


data = dde.data.PDE(
    geom,
    pde,
    bc,
    num_domain=nx_train ** 2,
    num_boundary=4 * nx_train,
    solution=func,
    num_test=nx_test ** 2,
)

net = dde.nn.FNN(
    [2] + [num_dense_nodes] * num_dense_layers + [1], activation, "Glorot uniform", task_name
)

new_save = False
for name, param in net.named_parameters():
    if os.path.exists(f"{log_dir}/{name}.npy"):
        continue
    new_save = True
    np.save(f"{log_dir}/{name}.npy", param.numpy())
    print(f"successfully save param {name} at [{log_dir}/{name}.npy]")

if new_save:
    print("第一次保存模型完毕，自动退出，请再次运行")
    exit(0)
else:
    print("所有模型参数均存在，开始训练...............")
if hard_constraint == True:
    net.apply_output_transform(transform)

model = dde.Model(data, net)

if hard_constraint == True:
    model.compile("adam", lr=learning_rate, metrics=["l2 relative error"])
else:
    loss_weights = [1, weights]
    model.compile(
        "adam",
        lr=learning_rate,
        metrics=["l2 relative error"],
        loss_weights=loss_weights,
    )

paddle.enable_static()
paddle.incubate.autograd.enable_prim()
losshistory, train_state = model.train(iterations=iterations)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
