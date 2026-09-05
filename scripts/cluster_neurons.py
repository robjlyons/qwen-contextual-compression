#!/usr/bin/env python
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import _bootstrap  # noqa: F401
from clustering.cluster_neurons import frequency_ordering,greedy_ordering,graph_component_ordering,signature_ordering,random_ordering,hybrid_ordering
from clustering.similarity import SparseGraph
from clustering.permutation import inverse_permutation
from evaluation.permutation_metrics import locality_metrics

def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument("--clustering-dir",type=Path,required=True); p.add_argument("--seed",type=int,default=42); p.add_argument("--group-size",type=int,default=64); p.add_argument("--hot-threshold",type=float,default=.75); p.add_argument("--cold-threshold",type=float,default=.10); a,_=p.parse_known_args(argv); d=a.clustering_dir
 graph=SparseGraph.load(d/"coactivation_graph.npz"); signatures=np.load(d/"selection_signatures.npy"); frequency=graph.node_frequency
 component,labels=graph_component_ordering(graph); signature,siglabels=signature_ordering(signatures,frequency)
 orders={"original":np.arange(graph.nodes),"frequency":frequency_ordering(frequency),"greedy":greedy_ordering(graph,a.group_size),"clustered":component,"signature":signature,"hybrid":hybrid_ordering(component,frequency,a.hot_threshold,a.cold_threshold),"random":random_ordering(graph.nodes,a.seed)}
 np.savez(d/"orderings.npz",**orders); np.save(d/"permutation.npy",orders["clustered"]); np.save(d/"inverse_permutation.npy",inverse_permutation(orders["clustered"]))
 pd.DataFrame({"neuron":np.arange(graph.nodes),"graph_cluster":labels,"signature_cluster":siglabels,"frequency":frequency,"temperature":np.where(frequency>a.hot_threshold,"HOT",np.where(frequency<a.cold_threshold,"COLD","WARM"))}).to_csv(d/"neuron_clusters.csv",index=False)
 rows=[]
 for name,order in orders.items(): rows.append({"ordering":name,**locality_metrics(graph,order)})
 pd.DataFrame(rows).to_csv(d/"locality_metrics.csv",index=False)
 print(pd.DataFrame(rows).to_string(index=False))
if __name__=="__main__": main()

