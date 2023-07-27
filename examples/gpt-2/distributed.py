from typing import List
import click
from tqdm import tqdm
import os

import torch

import hidet
from hidet import FlowGraph
from gpt_model import gpt2
from encoder import get_encoder

hidet.option.search_space(0)

bucket_size = 50


class GPT2Generator:
    def __init__(self, graph, max_num_tokens, use_fp16, model_name):
        import hidet.cuda.graph
        with hidet.graph.PassContext() as ctx:
            if use_fp16:
                ctx.set_precision('float16')
                ctx.set_mma('mma')
            graph = hidet.graph.optimize(graph)
        self.cuda_graph: hidet.cuda.graph.CudaGraph = graph.cuda_graph()
        self.encoder = get_encoder(model_name)
        self.max_num_tokens = max_num_tokens

        # get the torch view for the two hidet tensors
        self.input_ids = self.cuda_graph.inputs[0].torch()
        self.logits = self.cuda_graph.outputs[0].torch()

    def __call__(self, text: str=None) -> str:
        if text is not None:
            ids: List[int] = self.encoder.encode(text)
            num_init_tokens = len(ids)
            if num_init_tokens > self.max_num_tokens:
                return text
            self.input_ids[:num_init_tokens] = torch.asarray(ids)
            num_init_tokens_tensor = hidet.full([1], num_init_tokens, dtype=hidet.int32, device='cuda')
        else:
            num_init_tokens_tensor = hidet.empty([1], dtype=hidet.int32, device='cuda')

        hidet.distributed.broadcast(num_init_tokens_tensor, 0)
        hidet.distributed.broadcast(self.cuda_graph.inputs[0], 0)
        num_init_tokens = int(num_init_tokens_tensor[0])
        hidet.distributed.barrier()
        hidet.cuda.synchronize()

        for i in tqdm(range(num_init_tokens, self.max_num_tokens), "generating", ncols=80):
            self.cuda_graph.run()
            next_token: int = torch.argmax(self.logits[i - 1], dim=-1).item()
            self.input_ids[i] = next_token

        output_ids = self.input_ids[num_init_tokens:].cpu().tolist()
        output_text = self.encoder.decode(output_ids)
        return output_text

def run(world_size, rank, out_dir, max_num_tokens, use_fp16, model_name):
    hidet.cuda.set_device(rank)
    hidet.distributed.init_process_group(init_method=os.environ['INIT_METHOD'], world_size=world_size, rank=rank)
    flow_graph = hidet.distributed.load_partition(out_dir, rank)
    generator = GPT2Generator(flow_graph, max_num_tokens, use_fp16, model_name)
    while True:
        if rank == 0:
            x = click.prompt(">>> ", type=str, prompt_suffix="")
        else:
            x = None
        response = generator(x)
        if rank == 0:
            click.echo(x + response)

@click.command()
@click.option("--max-num-tokens", default=40, type=int, help='Max number of total tokens to process and generate',
              show_default=True)
@click.option("--use-fp16", is_flag=True, default=False, help='Use fp16', show_default=True)
@click.option("--model-size", default="124M", type=click.Choice(['124M', '355M', '774M', '1558M']), show_default=True)
@click.option("--tune", is_flag=True, default=False,
              help='Tune the operators for better performance. May take several minutes.', show_default=True)
@click.option("--out-dir", type=str, default='gpt2-parts')
@click.option("--recompile", is_flag=True, default=False)
def main(max_num_tokens: int, use_fp16: bool, model_size: str, tune: bool, out_dir: str, recompile:bool):
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    if recompile and rank == 0:
        if tune:
            hidet.option.search_space(2)
        graph: FlowGraph = gpt2(seq_length=max_num_tokens, model_size=model_size, use_fp16=use_fp16, opt=False)
        hidet.distributed.partition(graph, {'ngpus': world_size, 'mem_budget':  384 * 1024 * 1024, 'search_max_seconds': 300, 'solver_verbose': 2}, out_dir=out_dir)
    run(world_size, rank, out_dir, max_num_tokens, use_fp16, model_size)

if __name__ == "__main__":
    main()
