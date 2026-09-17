import numpy as np
import pytest
from clustering.permutation import inverse_permutation,remap_old_ids,validate_permutation

def test_inverse_and_id_remapping():
 perm=np.array([2,0,3,1]); inverse=inverse_permutation(perm)
 assert inverse.tolist()==[1,3,0,2]
 assert remap_old_ids(np.array([0,2]),perm).tolist()==[1,0]

def test_invalid_permutation_rejected():
 with pytest.raises(ValueError): validate_permutation([0,0,2])

