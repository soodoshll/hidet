# from typing import

from hidet.graph.ops.utils import Task, Operator, Tensor, TensorNode, compute, input_like


class AttenVarlenTask(Task):
    pass


class AttnVarlenOp(Operator):
    def __init__(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        cu_seqlens_q: Tensor,
        cu_seqlens_k: Tensor,
        cu_seqlens_v: Tensor,
        is_causal: bool = False,
    ):
        super().__init__(
            inputs=[q, k, v],
            task=AttenVarlenTask('attn_varlen', 
                                 input_like(q, 'q'), 
                                 input_like(k, 'k'), 
                                 input_like(v, 'v'), 
                                 input_like(cu_seqlens_q, 'cu_seqlens_q'),
                                 input_like(cu_seqlens_v, 'cu_seqlens_v'),
                                 input_like(cu_seqlens_k, 'cu_seqlens_k'),
                                 is_causal),
            attributes={'is_causal': is_causal},
        )


def attention_varlen(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    cu_seqlen_v: Tensor,
    is_causal: bool = False,
) -> Tensor:
    pass
