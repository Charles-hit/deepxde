"""Backend supported: tensorflow.compat.v1

Implementation of the Poisson 1D example in paper https://arxiv.org/abs/2012.10047.
References:
    https://github.com/PredictiveIntelligenceLab/MultiscalePINNs.
"""
import deepxde as dde
import numpy as np
# from deepxde.backend import tf
import paddle
import os
import random

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

A = 2
B = 50


def pde(x, y):
    dy_xx = dde.grad.hessian(y, x)
    # return (
    #     dy_xx
    #     + (np.pi * A) ** 2 * tf.sin(np.pi * A * x)
    #     + 0.1 * (np.pi * B) ** 2 * tf.sin(np.pi * B * x)
    # )
    return (
        dy_xx
        + (np.pi * A) ** 2 * paddle.sin(np.pi * A * x)
        + 0.1 * (np.pi * B) ** 2 * paddle.sin(np.pi * B * x)
    )


def func(x):
    return np.sin(np.pi * A * x) + 0.1 * np.sin(np.pi * B * x)


geom = dde.geometry.Interval(0, 1)
bc = dde.icbc.DirichletBC(geom, func, lambda _, on_boundary: on_boundary)
data = dde.data.PDE(
    geom,
    pde,
    bc,
    1280,
    2,
    train_distribution="pseudo",
    solution=func,
    num_test=10000,
)

layer_size = [1] + [100] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.MsFFN(layer_size, activation, initializer, sigmas=[1, 10])
 
model = dde.Model(data, net)
model.compile(
    "adam",
    lr=0.001,
    metrics=["l2 relative error"],
    decay=("inverse time", 2000, 0.9),
)

pde_residual_resampler = dde.callbacks.PDEResidualResampler(period=1)
model.train(iterations=20000, callbacks=[pde_residual_resampler])

dde.saveplot(model.losshistory, model.train_state, issave=True, isplot=True)
