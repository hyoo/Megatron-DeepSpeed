from megatron import fused_kernels
from argparse import Namespace

args = Namespace(rank=0, masked_softmax_fusion=True)
fused_kernels.load(args)
