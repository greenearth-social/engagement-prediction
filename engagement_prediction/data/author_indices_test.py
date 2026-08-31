from engagement_prediction.data.author_indices import AUTHOR_PAD_IDX, AUTHOR_UNK_IDX


def test_reserved_author_indices():
    assert AUTHOR_PAD_IDX == 0
    assert AUTHOR_UNK_IDX == 1
