#!/usr/bin/env python
import argparse,json
from pathlib import Path
import numpy as np
import torch,yaml
import _bootstrap  # noqa: F401
from extract.inspect_model import load_model
from extract.multilayer_capture import capture_multilayer,load_corpus
from transformers import AutoTokenizer

DEFAULT_LAYERS="0,8,16,24,32,40,48,56,63"
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--model",default="Qwen/Qwen3.8-27B");p.add_argument("--layers",default=DEFAULT_LAYERS);p.add_argument("--max-tokens-per-layer",type=int,default=2000);p.add_argument("--output-dir",type=Path,default=Path("results/multilayer"));p.add_argument("--input",type=Path);p.add_argument("--dataset");p.add_argument("--dataset-split",default="train");p.add_argument("--text-field",default="text");p.add_argument("--category-field",default="category");p.add_argument("--activation-dtype",choices=["fp16","bf16"],default="fp16");p.add_argument("--dtype",default="bfloat16");p.add_argument("--device-map",default="auto");p.add_argument("--offload-folder");p.add_argument("--cache-dir");p.add_argument("--revision");p.add_argument("--seed",type=int,default=42);p.add_argument("--include-special",action="store_true");p.add_argument("--all-layers",action="store_true")
 a,_=p.parse_known_args(argv);model=load_model(a.model,a.dtype,a.device_map,a.offload_folder,a.revision,full_model=True,cache_dir=a.cache_dir);backbone_config=getattr(model.config,"text_config",model.config);count=int(backbone_config.num_hidden_layers)
 requested=list(range(count)) if a.all_layers else [int(x) for x in a.layers.split(",")];default_requested=[int(x) for x in DEFAULT_LAYERS.split(",")]
 if requested==default_requested and count!=64: requested=np.linspace(0,count-1,9,dtype=int).tolist()
 if any(layer<0 or layer>=count for layer in requested): raise ValueError(f"Layers {requested} invalid for {count}-layer model")
 tokenizer=AutoTokenizer.from_pretrained(a.model,revision=a.revision,cache_dir=a.cache_dir);corpus=load_corpus(a.input,a.dataset,a.dataset_split,a.text_field,a.category_field);dtype=torch.float16 if a.activation_dtype=="fp16" else torch.bfloat16
 config={**vars(a),"layers":requested,"detected_layer_count":count};a.output_dir.mkdir(parents=True,exist_ok=True);(a.output_dir/"config.yaml").write_text(yaml.safe_dump(json.loads(json.dumps(config,default=str)),sort_keys=True));print(json.dumps(capture_multilayer(model,tokenizer,requested,corpus,a.output_dir,a.max_tokens_per_layer,dtype,not a.include_special,a.seed,a.revision),indent=2))
if __name__=="__main__":main()
