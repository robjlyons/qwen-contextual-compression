#!/usr/bin/env python
import argparse,hashlib,json
from pathlib import Path
import yaml
import _bootstrap  # noqa: F401
from transformers import AutoConfig,AutoTokenizer
from end_to_end.evaluation_runner import evaluate_schedules,generation_pairs
from end_to_end.schedules import load_schedules
from extract.inspect_model import load_model
from extract.multilayer_capture import estimate_corpus_tokens,load_corpus

DEFAULT_ORDER="dense,measured_conservative,measured_moderate,all_conservative,all_moderate"
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--model",default="Qwen/Qwen3.8-27B");p.add_argument("--input",type=Path);p.add_argument("--dataset");p.add_argument("--dataset-split",default="train");p.add_argument("--text-field",default="text");p.add_argument("--category-field",default="category");p.add_argument("--exclude-capture-metadata",type=Path);p.add_argument("--schedules",default=DEFAULT_ORDER);p.add_argument("--schedule-config",type=Path,default=Path("configs/end_to_end_schedules.yaml"));p.add_argument("--schedule-interpolation",choices=["nearest","linear"]);p.add_argument("--max-eval-tokens",type=int,default=250);p.add_argument("--device-map",default="auto");p.add_argument("--offload-folder");p.add_argument("--cache-dir");p.add_argument("--dtype",default="bfloat16");p.add_argument("--revision");p.add_argument("--norm-chunk-columns",type=int,default=256);p.add_argument("--output-dir",type=Path,default=Path("results/end_to_end"));p.add_argument("--force",action="store_true");p.add_argument("--generation",action="store_true");p.add_argument("--generation-prompts",type=int,default=20);p.add_argument("--max-new-tokens",type=int,default=64);a,_=p.parse_known_args(argv)
 if a.norm_chunk_columns<=0:p.error("--norm-chunk-columns must be positive")
 # Every cheap failure is resolved before full checkpoint loading.
 corpus=load_corpus(a.input,a.dataset,a.dataset_split,a.text_field,a.category_field);tokenizer=AutoTokenizer.from_pretrained(a.model,cache_dir=a.cache_dir,revision=a.revision)
 config=AutoConfig.from_pretrained(a.model,cache_dir=a.cache_dir,revision=a.revision);text_config=getattr(config,"text_config",config);layer_count=int(text_config.num_hidden_layers);names=[x for x in a.schedules.split(",") if x];schedules=load_schedules(a.schedule_config,layer_count,names,a.schedule_interpolation)
 overlap_status="not_checked"
 if a.exclude_capture_metadata:
  metadata=json.loads(a.exclude_capture_metadata.read_text());excluded=set(metadata.get("prompt_hashes",[]))
  if excluded:corpus=[item for item in corpus if hashlib.sha256(item["text"].encode()).hexdigest() not in excluded];overlap_status="excluded_by_exact_sha256"
  else:overlap_status="unknown: capture metadata has no prompt_hashes; no prompts were guessed or excluded"
 if not corpus:raise ValueError("No evaluation prompts remain after corpus loading/exclusion")
 diagnostics=estimate_corpus_tokens(tokenizer,corpus)
 if diagnostics["usable_content_tokens"]<a.max_eval_tokens:print(f"WARNING: corpus has only {diagnostics['usable_content_tokens']} usable tokens for requested {a.max_eval_tokens}")
 a.output_dir.mkdir(parents=True,exist_ok=True);config_record={**vars(a),"schedule_names":names,"layer_count":layer_count,"corpus_diagnostics":diagnostics,"calibration_overlap":overlap_status};(a.output_dir/"config.yaml").write_text(yaml.safe_dump(json.loads(json.dumps(config_record,default=str)),sort_keys=True));(a.output_dir/"schedule_expansions.json").write_text(json.dumps(schedules,indent=2)+"\n")
 model=load_model(a.model,a.dtype,a.device_map,a.offload_folder,a.revision,full_model=True,cache_dir=a.cache_dir);(a.output_dir/"model_metadata.json").write_text(json.dumps({"model":a.model,"revision":a.revision,"class":type(model).__name__,"layers":layer_count,"dtype":a.dtype,"device_map":getattr(model,"hf_device_map",None)},default=str,indent=2)+"\n");print(json.dumps(evaluate_schedules(model,tokenizer,corpus,schedules,a.output_dir,a.max_eval_tokens,a.force,a.norm_chunk_columns),indent=2))
 if a.generation:generation_pairs(model,tokenizer,corpus,schedules,a.output_dir,a.generation_prompts,a.max_new_tokens)
if __name__=="__main__":main()
