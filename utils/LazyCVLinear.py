import torch.nn.functional as F
import torch.nn as nn
from complextorch import CVTensor


class LazyCVLinear(nn.Module):
    r"""
    Complex-Valued Linear Layer
    ---------------------------

    Uses PyTorch's `LazyLinear` to infer the input size dynamically.
    """

    def __init__(self, out_features: int, bias: bool = False, device=None, dtype=None) -> None:
        super(LazyCVLinear, self).__init__()

        # Use LazyLinear to avoid specifying `in_features`
        self.linear_r = nn.LazyLinear(out_features=out_features, bias=bias, device=device, dtype=dtype)
        self.linear_i = nn.LazyLinear(out_features=out_features, bias=bias, device=device, dtype=dtype)

    @property
    def weight(self) -> CVTensor:
        return CVTensor(self.linear_r.weight, self.linear_i.weight)

    @property
    def bias(self) -> CVTensor:
        if self.linear_r.bias is None:
            return None
        else:
            return CVTensor(self.linear_r.bias, self.linear_i.bias)

    def forward(self, input: CVTensor) -> CVTensor:
        r"""
        Uses Gauss' trick to compute complex-valued multiplication efficiently.
        """
        t1 = self.linear_r(input.real)
        t2 = self.linear_i(input.imag)
        bias = (
            None
            if self.linear_r.bias is None
            else (self.linear_r.bias + self.linear_i.bias)
        )
        t3 = F.linear(
            input=(input.real + input.imag),
            weight=(self.linear_r.weight + self.linear_i.weight),
            bias=bias,
        )
        return CVTensor(t1 - t2, t3 - t2 - t1)

