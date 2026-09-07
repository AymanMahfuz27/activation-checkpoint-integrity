import torch
from torch.utils._python_dispatch import TorchDispatchMode
import torch.utils.checkpoint

class NoRandomnessMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        import pdb; pdb.set_trace()
        if func == torch.ops.aten.rand.default:
            return torch.ones(*args, **kwargs) / 2
        rs = func(*args, **kwargs)
        return rs

def f(x):
  y = torch.rand(x.shape)
  return x.clone() * y

def g(x):
  return torch.utils.checkpoint.checkpoint(f, x, use_reentrant=False)

x = torch.ones([], requires_grad=True)

with NoRandomnessMode():
  y = g(x)

y.backward()

# assertion failure
assert torch.allclose(x.grad, y)
