"""Scalable graph/signature orderings and a hot-dynamic-cold hybrid layout."""
from __future__ import annotations
import numpy as np
from clustering.permutation import validate_permutation
from clustering.similarity import SparseGraph


def frequency_ordering(frequency: np.ndarray) -> np.ndarray:
    return np.argsort(-frequency,kind="stable").astype(np.int64)


def random_ordering(neurons: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(neurons).astype(np.int64)


def _adjacency(graph: SparseGraph) -> list[list[tuple[int,float]]]:
    result=[[] for _ in range(graph.nodes)]
    for a,b,w in zip(graph.source,graph.target,graph.affinity):
        result[int(a)].append((int(b),float(w))); result[int(b)].append((int(a),float(w)))
    for row in result: row.sort(key=lambda item:(-item[1],item[0]))
    return result


def greedy_ordering(graph: SparseGraph, group_size: int = 64) -> np.ndarray:
    """Grow local groups by maximum affinity to the current group."""
    adjacency=_adjacency(graph); remaining=set(range(graph.nodes)); result=[]
    while remaining:
        seed=max(remaining,key=lambda n:(graph.node_frequency[n],-n)); group=[seed]; remaining.remove(seed)
        scores: dict[int,float]={}
        while remaining and len(group)<group_size:
            for neighbour,weight in adjacency[group[-1]]:
                if neighbour in remaining: scores[neighbour]=scores.get(neighbour,0)+weight
            candidate=max(scores,key=lambda n:(scores[n],graph.node_frequency[n],-n)) if scores else max(remaining,key=lambda n:(graph.node_frequency[n],-n))
            group.append(candidate); remaining.remove(candidate); scores.pop(candidate,None)
        result.extend(group)
    return validate_permutation(result,graph.nodes)


def graph_component_ordering(graph: SparseGraph) -> tuple[np.ndarray,np.ndarray]:
    """Connected-component graph clustering, ordered greedily within communities."""
    adjacency=_adjacency(graph); seen=np.zeros(graph.nodes,bool); components=[]
    for seed in np.argsort(-graph.node_frequency,kind="stable"):
        if seen[seed]: continue
        stack=[int(seed)]; seen[seed]=True; component=[]
        while stack:
            node=stack.pop(); component.append(node)
            for neighbour,_ in adjacency[node]:
                if not seen[neighbour]: seen[neighbour]=True; stack.append(neighbour)
        components.append(component)
    components.sort(key=lambda c:(-sum(graph.node_frequency[c]),-len(c)))
    order=[]; labels=np.empty(graph.nodes,np.int32)
    for label,component in enumerate(components):
        allowed=set(component); current=max(allowed,key=lambda n:(graph.node_frequency[n],-n))
        while allowed:
            if current not in allowed: current=max(allowed,key=lambda n:(graph.node_frequency[n],-n))
            order.append(current); labels[current]=label; allowed.remove(current)
            candidates=[(w,n) for n,w in adjacency[current] if n in allowed]
            current=max(candidates)[1] if candidates else -1
    return validate_permutation(order,graph.nodes),labels


def signature_ordering(signatures: np.ndarray, frequency: np.ndarray, bits: int = 20) -> tuple[np.ndarray,np.ndarray]:
    """Random-projection LSH clustering; avoids an N-by-N distance matrix."""
    bit_values=(signatures[:,:min(bits,signatures.shape[1])]>=0).astype(np.uint8)
    packed=np.packbits(bit_values,axis=1); keys=np.array([x.tobytes() for x in packed],dtype=f"S{packed.shape[1]}")
    # Frequency is a secondary key inside identical selection-signature buckets.
    order=np.lexsort((-frequency,keys)).astype(np.int64)
    _,labels=np.unique(keys,return_inverse=True)
    return order,labels.astype(np.int32)


def hybrid_ordering(clustered: np.ndarray, frequency: np.ndarray,
                    hot_threshold: float=.75, cold_threshold: float=.10) -> np.ndarray:
    hot=frequency>hot_threshold; cold=frequency<cold_threshold
    return np.concatenate([clustered[hot[clustered]],clustered[~hot[clustered]&~cold[clustered]],clustered[cold[clustered]]])

