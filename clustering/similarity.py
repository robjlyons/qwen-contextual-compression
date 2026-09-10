"""Sparse candidate graph and exact binary selection affinities."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class SparseGraph:
    source: np.ndarray
    target: np.ndarray
    affinity: np.ndarray
    intersections: np.ndarray
    node_frequency: np.ndarray
    nodes: int

    def save(self, path: Path) -> None:
        np.savez_compressed(path, source=self.source, target=self.target, affinity=self.affinity,
                            intersections=self.intersections, node_frequency=self.node_frequency,
                            nodes=np.array(self.nodes))

    @classmethod
    def load(cls, path: Path) -> "SparseGraph":
        data = np.load(path)
        return cls(data["source"], data["target"], data["affinity"], data["intersections"],
                   data["node_frequency"], int(data["nodes"]))


def binary_affinity(intersection: np.ndarray, left: np.ndarray, right: np.ndarray,
                    samples: int, metric: str) -> np.ndarray:
    intersection, left, right = intersection.astype(float), left.astype(float), right.astype(float)
    if metric == "raw": return intersection
    if metric == "jaccard": return intersection / np.maximum(left + right - intersection, 1)
    if metric == "cosine": return intersection / np.maximum(np.sqrt(left * right), 1)
    if metric == "conditional": return intersection / np.maximum(left, 1)
    if metric == "npmi":
        pxy=intersection/samples; px=left/samples; py=right/samples
        return np.where(pxy>0, np.log(pxy/np.maximum(px*py,1e-30))/-np.log(pxy), -1)
    raise ValueError(f"Unknown similarity {metric!r}")


def build_sparse_graph(packed: np.ndarray, samples: int, signatures: np.ndarray,
                       similarity: str = "jaccard", top_neighbours: int = 32,
                       min_affinity: float = 0.05, candidate_bucket: int = 128) -> SparseGraph:
    """LSH-sort signatures, then compute exact affinity only for bounded candidates."""
    nodes = len(packed); bits = min(20, signatures.shape[1])
    codes = np.packbits(signatures[:, :bits] >= 0, axis=1)
    keys = np.array([row.tobytes() for row in codes], dtype=f"S{codes.shape[1]}")
    order = np.argsort(keys, kind="stable")
    lut=np.array([int(i).bit_count() for i in range(256)],dtype=np.uint8)
    counts=lut[packed].sum(1,dtype=np.int64); edge_map: dict[tuple[int,int],float] = {}; inter_map={}
    # Neighbours in signature-sorted order form an approximate candidate set.
    radius=max(top_neighbours*2, candidate_bucket//2)
    for position,node in enumerate(order):
        candidates=order[max(0,position-radius):min(nodes,position+radius+1)]
        candidates=candidates[candidates!=node]
        approx=signatures[candidates]@signatures[node]
        candidates=candidates[np.argsort(-approx)[:top_neighbours*2]]
        intersections=lut[np.bitwise_and(packed[candidates],packed[node])].sum(1,dtype=np.int64)
        affinities=binary_affinity(intersections,np.full(len(candidates),counts[node]),counts[candidates],samples,similarity)
        for other,inter,aff in zip(candidates,intersections,affinities):
            if aff < min_affinity: continue
            edge=(min(int(node),int(other)),max(int(node),int(other)))
            if aff > edge_map.get(edge,-np.inf): edge_map[edge]=float(aff); inter_map[edge]=int(inter)
    # Enforce top-N after symmetrization.
    incident=[[] for _ in range(nodes)]
    for edge,aff in edge_map.items():
        incident[edge[0]].append((aff,edge)); incident[edge[1]].append((aff,edge))
    keep=set()
    for values in incident:
        keep.update(edge for _,edge in sorted(values,reverse=True)[:top_neighbours])
    edges=sorted(keep); source=np.array([x[0] for x in edges],np.int32); target=np.array([x[1] for x in edges],np.int32)
    affinity=np.array([edge_map[x] for x in edges],np.float32); intersections=np.array([inter_map[x] for x in edges],np.int32)
    return SparseGraph(source,target,affinity,intersections,counts/samples,nodes)


def graph_statistics(graph: SparseGraph) -> dict:
    degree=np.bincount(np.concatenate([graph.source,graph.target]),minlength=graph.nodes)
    parent=np.arange(graph.nodes)
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    for a,b in zip(graph.source,graph.target):
        ra,rb=find(a),find(b)
        if ra!=rb: parent[rb]=ra
    components=len({find(i) for i in range(graph.nodes)})
    degree_hist=np.bincount(degree); affinity_hist,affinity_bins=np.histogram(graph.affinity,bins=20,range=(0,1))
    return {"nodes":graph.nodes,"edges":len(graph.source),"average_degree":float(degree.mean()),
            "connected_components":components,"isolated_nodes":int((degree==0).sum()),
            "degree":{"min":int(degree.min()),"median":float(np.median(degree)),"max":int(degree.max())},
            "degree_distribution":{"degree":list(range(len(degree_hist))),"node_count":degree_hist.tolist()},
            "affinity":{"min":float(graph.affinity.min()) if len(graph.affinity) else 0,
                        "median":float(np.median(graph.affinity)) if len(graph.affinity) else 0,
                        "max":float(graph.affinity.max()) if len(graph.affinity) else 0},
            "affinity_distribution":{"bin_edges":affinity_bins.tolist(),"edge_count":affinity_hist.tolist()}}
