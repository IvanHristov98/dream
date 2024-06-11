from typing import Callable, List
import math

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from dream import model


EncodeCaptionFn = Callable[[str], List[np.ndarray]]

MAX_N = 20
NS = [1, 5, 10, MAX_N]


def relevance_score_of_codes(ys: List[np.ndarray], x: np.ndarray) -> float:
    similarities = cosine_similarity([x], ys)

    return max_similarity(similarities)


def max_similarity(similarities: np.ndarray) -> float:
    # Negative relevance scores are possible and having such should be interpreted as 0 so that
    # relevant image retrievals aren't penalized.
    return max(similarities.max(), 0)


def im_caption_codes(im: model.Image, encode_caption_fn: EncodeCaptionFn) -> List[np.ndarray]:
    ys = []

    for im_caption in im.captions:
        # there should always be exactly one code
        code = encode_caption_fn(im_caption)[0]
        ys.append(code)

    return ys


def precision(rel_scores: List[float]) -> float:
    total = 0.0

    for rel_score in rel_scores:
        total += rel_score

    return total / len(rel_scores)


def ndcg(rel_scores: List[float]) -> float:
    dcg = 0.0

    for i, rel_score in enumerate(rel_scores):
        # plus 2 instead of 1 because in the original formula i should start from 1
        dcg += rel_score / math.log2(i + 2)

    sorted_rel_scores = rel_scores.copy()
    sorted_rel_scores.sort(reverse=True)

    idcg = 0.0

    for i, sorted_rel_score in enumerate(sorted_rel_scores):
        idcg += sorted_rel_score / math.log2(i + 2)

    if idcg == 0:
        return 0

    return dcg / idcg
