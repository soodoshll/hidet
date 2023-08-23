from typing import Sequence
from hidet import ir
from hidet.ir import primitives as prim
from hidet.ir.compute import reduce
from hidet.ir.expr import  if_then_else
from hidet.graph.ops.utils import Task, Operator, Tensor, TensorNode, compute, input_like
from hidet.graph.ops.utils import broadcast_shape, broadcast_shapes, broadcast_indices

class AttenVarlenTask(Task):
    def __init__(
        self,
        name: str,
        q: TensorNode,
        k: TensorNode,
        v: TensorNode,
        cu_seqlens_q: Sequence[int],
        cu_seqlens_k: Sequence[int],
        is_causal: bool = False,
    ):
        assert len(q.shape) == 3
        assert len(k.shape) == 3
        assert len(v.shape) == 3
        assert len(cu_seqlens_q) == len(cu_seqlens_k)
        n_seq = len(cu_seqlens_q)
        assert n_seq > 0

        q_shape = q.shape
        k_shape = k.shape
        v_shape = v.shape

        self._assert( 
            ir.logical_and(q_shape[0] == k_shape[0] == v_shape[0]),
            f"q/k/v should have same number of heads, but got {q_shape[1]} vs {k_shape[2]} vs {v_shape[1]}"
        )
        self._assert(
            ir.logical_and(q_shape[2] == k_shape[1] == v_shape[2]),
            f"q/k/v should have same hidden dimension, but got {q_shape[2]} vs {k_shape[1]} vs {v_shape[2]}"
        )
        self._assert(
            ir.logical_and(k_shape[2] == v_shape[1]),
            f"k and v should have same number of tokens, but got {k_shape[0]} vs {v_shape[0]}"
        )

        num_heads = q_shape[0]
        d_size = q_shape[2]
        q_start, k_start = 0, 0
        o_list = []
        for i in range(n_seq):
            seqlen_q = cu_seqlens_q[i]
            seqlen_k = cu_seqlens_k[i]
            qk = compute(
                name=f'qk_{i}',
                shape=[num_heads, seqlen_q, seqlen_k],
                fcompute=lambda *indices: reduce(
                    shape=[d_size],
                    fcompute=lambda d: q[indices[0], indices[1] + q_start, d] * k[indices[0], d, indices[2] + k_start],
                    reduce_type='sum',
                ),
            )
            
            max_value = compute(
                name=f'max_value_{i}',
                shape=[num_heads, seqlen_q],
                fcompute=lambda *indices: reduce(
                    shape=[seqlen_k], fcompute=lambda a: qk[indices[:2] + (a,)], reduce_type='max'
                ),
            )

            exp_value = compute(
                name=f'exp_value_{i}',
                shape=(num_heads, seqlen_q, seqlen_k),
                fcompute=lambda *indices: prim.exp(qk[indices] - max_value[indices[:2]]),
            )

            sum_value = compute(
                name=f'sum_value_{i}',
                shape=[num_heads, seqlen_q],
                fcompute=lambda *indices: reduce(
                    shape=[seqlen_k],
                    fcompute=lambda a: exp_value[indices + (a, )],
                    reduce_type='sum',
                ),
            )

            sm = compute(
                name=f'sm_{i}',
                shape=(num_heads, seqlen_q, seqlen_k),
                fcompute=lambda *indices: exp_value[indices] / sum_value[indices[:2]],
            )
            
            o = compute(
                name=f'o_{i}',
                shape=[num_heads, seqlen_q, d_size],
                fcompute=lambda *indices: reduce(
                    shape=[seqlen_k],
                    fcompute=lambda a: sm[indices[:2] + (a,)] * v[indices[0], a + k_start, indices[2]],
                    reduce_type='sum',
                ),
            )
            o_list.append(o)
            q_start = q_start + cu_seqlens_q[i]
            k_start = k_start + cu_seqlens_k[i]
        
        # concatenation
        dtype = o_list[0].type.dtype
        def fmap(head, token, dim):
            shapes = [o.shape[1] for o in o_list]
            pre_sum = [sum(shapes[:i]) for i in range(n_seq)]
            tot_len = sum(shapes)
            value = o_list[0][head, token, dim]
            for seq_id, seq_start in enumerate(pre_sum):
                if seq_id == 0:
                    continue
                value = if_then_else(token >= seq_start, o_list[seq_id][head, token - seq_start, dim], value)
            value = if_then_else(token >= tot_len, dtype.zero, value)
            return value


        o_concat = compute(
            name='o_concat',
            shape=q.shape,
            fcompute=lambda *indices: fmap(*indices)
        )
        super().__init__(
            name='atten_varlen',
            inputs=[q, k, v],
            outputs=[o_concat],
            attributes={'is_causal': is_causal, 'cu_seqlens_q': cu_seqlens_q, 'cu_seqlens_k': cu_seqlens_k},
        )


class AttnVarlenOp(Operator):
    def __init__(
        self, q: Tensor, k: Tensor, v: Tensor, cu_seqlens_q: Sequence[int], cu_seqlens_k: Sequence[int], is_causal: bool = False
    ):
        super().__init__(
            inputs=[q, k, v],
            task=AttenVarlenTask(
                'attn_varlen',
                input_like(q, 'q'),
                input_like(k, 'k'),
                input_like(v, 'v'),
                cu_seqlens_q,
                cu_seqlens_k,
                is_causal,
            ),
            attributes={'is_causal': is_causal, 'cu_seqlens_q': cu_seqlens_q, 'cu_seqlens_k': cu_seqlens_k},
        )


def attention_varlen(
    q: Tensor, k: Tensor, v: Tensor, seqlens_q: Sequence[int], seqlens_k: Sequence[int], is_causal: bool = False
) -> Tensor:
    return AttnVarlenOp(q, k, v, seqlens_q, seqlens_k, is_causal).outputs[0]
