import numpy as np
import torch
from clustering.block_layout import select_blocks
from clustering.cluster_neurons import frequency_ordering,signature_ordering
from evaluation.permutation_metrics import locality_metrics
from clustering.similarity import SparseGraph

def test_equal_budget_selects_complete_blocks():
 scores=torch.tensor([[9.,8.,1.,1.,7.,6.,0.,0.]])
 oracle=torch.tensor([[0,1,4]])
 selected=select_blocks(scores,oracle,2,"equal_budget",3,.99)
 assert selected.block_count.item()==2 and selected.loaded_count.item()==4
 ids=selected.indices[0,selected.valid_mask[0]].tolist(); assert ids==[0,1,4,5]

def test_equal_coverage_reports_expansion_and_recall():
 scores=torch.tensor([[9.,1.,8.,1.,7.,1.,0.,0.]]); oracle=torch.tensor([[0,2,4]])
 selected=select_blocks(scores,oracle,2,"equal_coverage",3,1.)
 assert selected.oracle_recall.item()==1 and selected.loaded_count.item()==6
 assert selected.expansion.item()==2

def test_signature_and_frequency_orderings_are_bijections():
 signatures=np.array([[1,1],[-1,-1],[1,-1]],np.float32); freq=np.array([.2,.9,.4])
 order,_=signature_ordering(signatures,freq); assert sorted(order.tolist())==[0,1,2]
 assert frequency_ordering(freq).tolist()==[1,2,0]

def test_locality_rewards_adjacent_affinity_pair():
 graph=SparseGraph(np.array([0]),np.array([3]),np.array([1.],np.float32),np.array([2]),np.ones(4),4)
 original=locality_metrics(graph,np.arange(4)); reordered=locality_metrics(graph,np.array([0,3,1,2]))
 assert reordered["weighted_mean_distance"]<original["weighted_mean_distance"]
