import time
from typing import List, Callable, NamedTuple, Dict
import uuid

from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from dream.experiment import fixture
from dream.experiment import model as experimentmodel
from dream.semsearch import service as semsearchservice
from dream import model


_LABELED_ONLY = "labeled_only"


class ExperimentResult(NamedTuple):
    mean_precision: float
    mean_ndcg: float
    mean_duration_sec: float


def run_semsearch_experiments(
    semsearch_svc: semsearchservice.SemSearchService,
    encode_caption_fn: experimentmodel.EncodeCaptionFn,
) -> Dict[str, ExperimentResult]:
    results = {}

    for n in experimentmodel.NS:
        results[f"{_LABELED_ONLY}_{n}"] = _run_semsearch_experiment(
            semsearch_svc.query_labeled_ims,
            semsearch_svc.get_im_metadata,
            n,
            encode_caption_fn,
        )

    return results


def _run_semsearch_experiment(
    search_fn: Callable[[str, int], List[uuid.UUID]],
    get_im_metadata_fn: Callable[[uuid.UUID], model.Image],
    n: int,
    encode_caption_fn: experimentmodel.EncodeCaptionFn,
) -> ExperimentResult:
    total_precision = 0.0
    total_ndcg = 0.0
    total_duration_sec = 0.0
    test_captions_count = 0

    test_captions = list(fixture.ideal_caption_relevances.keys())

    for i in tqdm(range(len(test_captions))):
        test_caption = test_captions[i]

        start = time.time()
        im_ids = search_fn(test_caption, n)
        total_duration_sec += time.time() - start

        rel_scores = []

        for rank, im_id in enumerate(im_ids):
            im = get_im_metadata_fn(im_id)
            rel_score = relevance_score(im, rank, test_caption, encode_caption_fn)

            rel_scores.append(rel_score)

        total_precision += experimentmodel.precision(rel_scores)
        total_ndcg += experimentmodel.ndcg(rel_scores)

        test_captions_count += 1

    return ExperimentResult(
        mean_precision=total_precision / test_captions_count,
        mean_ndcg=total_ndcg / test_captions_count,
        mean_duration_sec=total_duration_sec / test_captions_count,
    )


def relevance_score(
    im: model.Image, rank: int, caption: str, encode_caption_fn: experimentmodel.EncodeCaptionFn
) -> float:
    xs = encode_caption_fn(caption)
    ys = experimentmodel.im_caption_codes(im, encode_caption_fn)

    similarities = cosine_similarity(xs, ys)

    ideal_similarity = fixture.ideal_caption_relevances[caption][rank]

    if ideal_similarity <= 0:
        return 0.0

    return experimentmodel.max_similarity(similarities) / ideal_similarity
