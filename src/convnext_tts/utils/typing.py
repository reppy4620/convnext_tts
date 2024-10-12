# https://github.com/lucidrains/alphafold3-pytorch/blob/4768a65e1fd2556e106758921004f5d45bae4ea1/alphafold3_pytorch/tensor_typing.py#L47-L57

from jaxtyping import Bool, Float, Int, Shaped
from torch import Tensor


class TorchTyping:
    def __init__(self, abstract_dtype):
        self.abstract_dtype = abstract_dtype

    def __getitem__(self, shapes: str):
        return self.abstract_dtype[Tensor, shapes]


Shaped = TorchTyping(Shaped)
Float = TorchTyping(Float)
Int = TorchTyping(Int)
Bool = TorchTyping(Bool)

__all__ = ["Shaped", "Float", "Int", "Bool"]
