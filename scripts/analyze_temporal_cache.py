#!/usr/bin/env python
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
import torch
import _bootstrap
from runtime.selector import RuntimeSelector
from runtime.temporal import recover_sequence_metadata,simulate_lru_cache,temporal_mask_statistics
from runtime.diagnostics import hardware_metadata
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--results-dir",type=Path,required=True);p.add_argument("--layer",type=int,default=0);p.add_argument("--stage1-run",required=True);p.add_argument("--retention",type=float,default=.5);p.add_argument("--device",default="cuda");a=p.parse_args(argv);layer=a.results_dir/f"layer_{a.layer:03d}";target=layer/"targets";metadata=recover_sequence_metadata(target/"sample_metadata.jsonl");stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ");out=a.results_dir.parent/"runtime"/f"phase5b_temporal_{stamp}";out.mkdir(parents=True)
 if metadata is None:
  result={"status":"temporal cache analysis unsupported from current saved metadata"};(out/"temporal_stats.json").write_text(json.dumps(result,indent=2)+"\n");(out/"hardware.json").write_text(json.dumps(hardware_metadata(),indent=2)+"\n");(out/"report.md").write_text("# Phase 5B temporal analysis\n\n"+result["status"]+"\n");print(out);return
 data=torch.load(target/"targets.pt",map_location="cpu",weights_only=True);selector=RuntimeSelector.from_checkpoint(layer/a.stage1_run/"best.pt",a.retention,a.device);masks=selector.select(data["inputs"].to(a.device)).cpu();neuron_count=selector.predictor.neurons.out_features;stats=temporal_mask_statistics(masks,metadata);caches={str(value):simulate_lru_cache(masks,metadata,value,neuron_count) for value in (.5,.55,.6,.65,.75,1.)};config={"layer":a.layer,"retention":a.retention,"stage1_run":a.stage1_run,"adjacency":"same prompt and consecutive token_position only","cache_reset":"prompt-local","pcie_bandwidth_GBps":13.};(out/"config.json").write_text(json.dumps(config,indent=2)+"\n");(out/"hardware.json").write_text(json.dumps(hardware_metadata(),indent=2)+"\n");(out/"temporal_stats.json").write_text(json.dumps(stats,indent=2)+"\n");(out/"cache_simulation.json").write_text(json.dumps(caches,indent=2)+"\n");(out/"report.md").write_text("# Phase 5B temporal cache analysis\n\nPrompt-safe adjacency only. PCIe refill times are estimates, not overlapped runtime.\n\n```json\n"+json.dumps({"temporal":stats,"cache":caches},indent=2)+"\n```\n");print(out)
if __name__=="__main__":main()
