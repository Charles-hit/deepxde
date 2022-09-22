"""paddle backend implementation"""
from distutils.version import LooseVersion

import paddle
import random 
import numpy

if LooseVersion(paddle.__version__) < LooseVersion("2.3.0") and LooseVersion(paddle.__version__) != LooseVersion("0.0.0") :
    raise RuntimeError("DeepXDE requires PaddlePaddle>=2.3.0")

if paddle.device.is_compiled_with_cuda():
    paddle.device.set_device("gpu")

lib = paddle


def data_type_dict():
    return {
        "float16": paddle.float16,
        "float32": paddle.float32,
        "float64": paddle.float64,
        "uint8": paddle.uint8,
        "int8": paddle.int8,
        "int16": paddle.int16,
        "int32": paddle.int32,
        "int64": paddle.int64,
        "bool": paddle.bool,
    }


def is_tensor(obj):
    return paddle.is_tensor(obj)


def shape(input_tensor):
    return input_tensor.shape


def ndim(input_tensor):
    return input_tensor.ndim


def transpose(tensor, axes=None):
    if axes is None:
        axes = tuple(range(tensor.ndim)[::-1])
    return paddle.transpose(tensor, axes)


def reshape(tensor, shape):
    return paddle.reshape(tensor, shape)


def Variable(initial_value, dtype=None):
    if paddle.in_dynamic_mode():
        return paddle.to_tensor(initial_value, dtype=dtype, stop_gradient=False)
    else:
        return paddle.static.create_parameter(shape=[1], dtype='float32', default_initializer=paddle.nn.initializer.Constant(value=initial_value))


def as_tensor(data, dtype=None):
    if paddle.is_tensor(data):
        if dtype is None or data.dtype == dtype:
            return data
        return data.astype(dtype)
    return paddle.to_tensor(data, dtype=dtype)


def from_numpy(np_array):
    return paddle.to_tensor(np_array)


def to_numpy(input_tensor):
    return input_tensor.detach().cpu().numpy()


def elu(x):
    return paddle.nn.functional.elu(x)


def relu(x):
    return paddle.nn.functional.relu(x)


def selu(x):
    return paddle.nn.functional.selu(x)


def sigmoid(x):
    return paddle.nn.functional.sigmoid(x)


def silu(x):
    return paddle.nn.functional.silu(x)


def sin(x):
    return paddle.sin(x)


def square(x):
    if paddle.in_dynamic_mode():
        return paddle.square(x)
    if paddle.incubate.autograd.prim_enabled():
        return x * x
    else:
        return paddle.square(x)
        # return paddle.pow(x, paddle.full_like(x, 2.0, x.dtype))

def norm(x, p=None, axis=None, keepdims=False):
    return paddle.linalg.norm(x, p=p, axis=axis, keepdim=keepdims)


def tanh(x):
    return paddle.tanh(x)


def mean(input_tensor, dim, keepdims=False):
    return paddle.mean(input_tensor, axis=dim, keepdim=keepdims)


def reduce_mean(input_tensor):
    return paddle.mean(input_tensor)


def sum(input_tensor, dim, keepdims=False):
    return paddle.sum(input_tensor, axis=dim, keepdim=keepdims)


def reduce_sum(input_tensor):
    return paddle.sum(input_tensor)


def zeros(shape, dtype):
    return paddle.zeros(shape, dtype=dtype)

def zeros_like(input_tensor):
    if paddle.in_dynamic_mode():
        return paddle.full_like(input_tensor, 0.0)
    if paddle.incubate.autograd.prim_enabled():
        # This ugly trick should be fixed when we support fill_any_like in prim.
        return paddle.full(input_tensor.shape, 0.0, input_tensor.dtype)
    else:
        return paddle.full_like(input_tensor, 0.0)
    #return paddle.zeros(input_tensor.shape,input_tensor.dtype)

def control_seed(number):
    paddle.seed(number)
    numpy.random.seed(number)
    random.seed(number)
