import pytest
import torch

from integration.freetoken_qcc.low_vram.embedding import HostBackedEmbedding


@pytest.mark.parametrize("ids", [torch.tensor([0]), torch.tensor([0, 4, 9]), torch.tensor([3, 3, 3]), torch.tensor([0, 9])])
def test_host_embedding_matches_resident_lookup(ids):
    weight = torch.arange(50, dtype=torch.float16).reshape(10, 5)
    lookup = HostBackedEmbedding(weight, "cpu")
    torch.testing.assert_close(lookup(ids), torch.nn.functional.embedding(ids, weight), rtol=0, atol=0)


def test_host_embedding_rejects_out_of_range():
    lookup = HostBackedEmbedding(torch.zeros(4, 3), "cpu")
    with pytest.raises(IndexError):
        lookup(torch.tensor([4]))
