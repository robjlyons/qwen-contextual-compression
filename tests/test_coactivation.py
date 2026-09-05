import numpy as np
from clustering.coactivation import packed_sample_prefix,selection_frequencies,unpack_neurons
from clustering.similarity import binary_affinity,build_sparse_graph

def _packed(mask): return np.packbits(mask,axis=1,bitorder="little")

def test_packed_masks_and_calibration_prefix_do_not_leak_validation():
 mask=np.array([[1,1,0,0,1],[1,0,1,0,1],[0,0,0,1,1]],dtype=np.uint8); packed=_packed(mask)
 prefix=packed_sample_prefix(packed,4)
 assert unpack_neurons(prefix,0,3,4).astype(int).tolist()==mask[:,:4].tolist()
 assert np.allclose(selection_frequencies(prefix,4),[.5,.5,.25])

def test_binary_similarities():
 inter=np.array([2]); left=np.array([4]); right=np.array([3])
 assert np.allclose(binary_affinity(inter,left,right,10,"jaccard"),[.4])
 assert np.allclose(binary_affinity(inter,left,right,10,"conditional"),[.5])

def test_sparse_graph_never_materialises_dense_pair_matrix():
 mask=np.array([[1,1,0,0],[1,1,0,0],[0,0,1,1]],dtype=np.uint8); packed=_packed(mask)
 signatures=np.array([[1.,0.],[1.,0.],[0.,1.]],np.float32)
 graph=build_sparse_graph(packed,4,signatures,"jaccard",top_neighbours=2,min_affinity=.1)
 assert any({int(a),int(b)}=={0,1} and np.isclose(w,1) for a,b,w in zip(graph.source,graph.target,graph.affinity))

