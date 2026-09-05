from __future__ import annotations
import numpy as np
import pandas as pd


class NeuronAccumulator:
    def __init__(self, width: int, rank_retention: float = .30):
        self.width=width; self.k=max(1, round(width*rank_retention)); self.samples=0
        self.count=np.zeros(width,np.int64); self.rank_sum=np.zeros(width); self.ranks=[[] for _ in range(width)]
        self.abs_sum=np.zeros(width); self.weighted_sum=np.zeros(width)
    def update(self, scores, activations, weighted):
        ranks=np.argsort(-scores,axis=1); top=ranks[:,:self.k]
        for row in ranks:
            inverse=np.empty(self.width,np.int32); inverse[row]=np.arange(1,self.width+1)
            self.rank_sum += inverse
            for j in row[:self.k]: self.ranks[j].append(int(inverse[j]))
        np.add.at(self.count,top.ravel(),1); self.abs_sum += np.abs(activations).sum(0); self.weighted_sum += weighted.sum(0); self.samples += len(scores)
    def frame(self):
        med=np.array([np.median(x) if x else np.nan for x in self.ranks])
        return pd.DataFrame({"neuron":np.arange(self.width),"selection_count":self.count,
          "selection_frequency":self.count/max(1,self.samples),"mean_rank":self.rank_sum/max(1,self.samples),
          "median_selected_rank":med,"mean_activation_magnitude":self.abs_sum/max(1,self.samples),
          "mean_weighted_contribution":self.weighted_sum/max(1,self.samples)})


def hot_coverage(stats: pd.DataFrame) -> pd.DataFrame:
    values=np.sort(stats.selection_count.to_numpy())[::-1]; total=max(1,values.sum())
    return pd.DataFrame({"hot_neurons":np.arange(1,len(values)+1),"hot_fraction":np.arange(1,len(values)+1)/len(values),
                         "selection_coverage":np.cumsum(values)/total})

