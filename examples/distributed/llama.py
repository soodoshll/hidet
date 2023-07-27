import argparse 

from transformers import LlamaConfig, LlamaTokenizer
from transformers import LlamaForCausalLM as hfLm

import hidet
import hidet.testing

from hidet.testing.models.llama import LlamaForCausalLM

import time
import os
import numpy

def run(world_size, rank, out_dir, config):
    hidet.cuda.set_device(rank)
    hidet.distributed.init_process_group(init_method=os.environ['INIT_METHOD'], world_size=world_size, rank=rank)
    flow_graph = hidet.distributed.load_partition(out_dir, rank)
    x = hidet.ones([1, 64], device='cuda', dtype=hidet.int32)
    position_ids = hidet.arange(0, config.max_position_embeddings, dtype=hidet.int32, device='cuda').unsqueeze(0)
    compiled = flow_graph.build()
    print(compiled(x, position_ids))

def build_flow_graph(model, batch_size=1, seq_length=64, device='cpu', dtype='float16'):
    config = model.config
    input_ids = hidet.symbol([batch_size, seq_length], dtype=hidet.int32, device=device)
    position_ids = hidet.symbol([batch_size, config.max_position_embeddings], dtype=hidet.int32, device=device)

    y = model(input_ids, position_ids=position_ids, past_key_values=None)
    outputs = [y['past_key_values'][-1][0]]
    inputs = [input_ids, position_ids]
    return hidet.trace_from(outputs, inputs)

parser = argparse.ArgumentParser()
parser.add_argument("--recompile", action='store_true')
args = parser.parse_args()
world_size = int(os.environ['WORLD_SIZE'])
rank = int(os.environ['RANK'])
config = LlamaConfig(
    # num_hidden_layers=4
)

if args.recompile and rank == 0:
    print("Loading model...")
    load_model_start = time.time()
    model = LlamaForCausalLM(config)
    model.to('float16')
    flow_graph = build_flow_graph(model, device='cpu')
    print(f"Model loaded, {time.time() - load_model_start : .2f}s")

    print("Start partitioning")
    partition_start = time.time()
    hidet.distributed.partition(flow_graph, {'ngpus': world_size, 'mem_budget': 10 * 1024 * 1024 * 1024, 'search_max_seconds': 300}, 'llama-parts')
    print(f"Partitioning finished, {time.time() - partition_start: .2f}s")

run(world_size, rank, 'llama-parts', config)
