# Disable auto load flagcx when load flagcx to
# avoid recursive init when only import flagcx without
# import torch
import os
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
import torch
os.environ.pop('TORCH_DEVICE_BACKEND_AUTOLOAD')
from functools import wraps
import torch.distributed as dist
from torch.distributed.distributed_c10d import _coalescing_manager

from ._C import *

def init():
    pass

def replace_prefix(arg):
    device_list = ["cuda", "mlu", "npu", "musa", "txda"]
    flagcx_prefix = "flagcx_dev"
    if isinstance(arg, str):
        for string in device_list:
            if string in arg:
                arg = arg.replace(string, flagcx_prefix)
    return arg

def replace_device_args(fn):
    @wraps(fn)
    def wrapper_fn(*args, **kwargs):
        if args:
            args = list(args)
            args[1]=replace_prefix(args[1])
        return fn(*args, **kwargs)
    return wrapper_fn

def replace_device(module, fn_name):
    fn = getattr(module, fn_name)
    if fn:
        setattr(module, fn_name, replace_device_args(fn))

def _batch_isend_irecv(p2p_op_list):
    if not isinstance(p2p_op_list, list) or not all(
        isinstance(p2p_op, dist.P2POp) for p2p_op in p2p_op_list
    ):
        raise ValueError(
            "Invalid ``p2p_op_list``. Each op is expected to "
            "to be of type ``torch.distributed.P2POp``."
        )

    group = p2p_op_list[0].group
    device = p2p_op_list[0].tensor.device
    if not all(group == p2p_op.group for p2p_op in p2p_op_list):
        raise ValueError("All ops need to use the same group.")

    # we adopt the implementation of PyTorch 2.5
    with _coalescing_manager(group, device, async_ops=True) as cm:
        for p2p_op in p2p_op_list:
            p2p_op.op(
                p2p_op.tensor,
                p2p_op.peer,
                p2p_op.group,
                p2p_op.tag
            )
    return cm.works

# A lightweight hook to support batch_isend_irecv in PyTorch 2.7 and newer
torch.distributed.batch_isend_irecv = _batch_isend_irecv
replace_device(dist.distributed_c10d.PrefixStore, "__init__")
