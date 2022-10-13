"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import sys
# Import tf if using backend tensorflow.compat.v1 or tensorflow
# from deepxde.backend import tf
# Import torch if using backend pytorch
import torch
# Import paddle if using backend paddle
import paddle
# paddle.enable_static()
# paddle.incubate.autograd.enable_prim()
from deepxde.backend import backend_name, tf, torch, jax, paddle
from deepxde import backend as bkd
bkd.control_seed(100)

def pde(x, y):
    dy_xx = dde.grad.hessian(y, x)
    return -dy_xx - np.pi ** 2 * bkd.sin(np.pi * x)


def boundary(x, on_boundary):
    return on_boundary


def func(x):
    return np.sin(np.pi * x)

geom = dde.geometry.Interval(-1, 1)
bc = dde.icbc.DirichletBC(geom, func, boundary)
data = dde.data.PDE(geom, pde, bc, 16, 2, solution=func, num_test=100)

layer_size = [1] + [50] * 3 + [1]
# set the same initializer param #
# paddle init param
w_array = []
input_str = []
file_name1 = sys.argv[1]
with open(file_name1, mode='r') as f1:
    for line in f1:
        input_str.append(line)
print("input_str.size: ", len(input_str))
j = 0
for i in range(1, len(layer_size)):
    shape = (layer_size[i-1], layer_size[i])
    w_line = input_str[j]
    w = []
    tmp = w_line.split(',')
    for num in tmp:
        w.append(np.float(num))
    w = np.array(w).reshape(shape)
    print("w . shape :", w.shape)
    j = j+2
    w_array.append(w)
###############################

activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer, w_array)

model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])

losshistory, train_state = model.train(iterations=10000)
# Optional: Save the model during training.
# checkpointer = dde.callbacks.ModelCheckpoint(
#     "model/model", verbose=1, save_better_only=True
# )
# Optional: Save the movie of the network solution during training.
# ImageMagick (https://imagemagick.org/) is required to generate the movie.
# movie = dde.callbacks.MovieDumper(
#     "model/movie", [-1], [1], period=100, save_spectrum=True, y_reference=func
# )
# losshistory, train_state = model.train(iterations=10000, callbacks=[checkpointer, movie])

dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# Optional: Restore the saved model with the smallest training loss
# model.restore(f"model/model-{train_state.best_step}.ckpt", verbose=1)
# Plot PDE residual
x = geom.uniform_points(1000, True)
# y_ = func(x)
# file_name_y_ = 'standard_y'
# with open(file_name_y_,'w') as f:
#     np.savetxt(f,y_,delimiter=",")
    
# y = model.predict(x, operator=pde)
y = model.predict(x, operator=None)

if backend_name == 'paddle':
    file_namex = 'paddle_x'
    file_namey = 'paddle_y'
elif backend_name == 'pytorch':
    file_namex = 'pytorch_x'
    file_namey = 'pytorch_y'
elif backend_name == 'tensorflow':
    file_namex = 'tensorflow_x'
    file_namey = 'tensorflow_y'

    
with open(file_namex,'ab') as f:
    np.savetxt(f,x,delimiter=",")
with open(file_namey,'ab') as g:
    np.savetxt(g,y,delimiter=",")

# plt.figure()
# plt.plot(x, y)
# plt.xlabel("x")
# plt.ylabel("PDE residual")
# plt.show()
