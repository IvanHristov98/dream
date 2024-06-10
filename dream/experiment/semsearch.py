import time
from typing import List, Callable, NamedTuple, Dict
import uuid
from tqdm import tqdm

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

    for i in tqdm(range(len(fixture.captions))):
        caption = fixture.captions[i]

        start = time.time()
        im_ids = search_fn(caption, n)
        total_duration_sec += time.time() - start

        rel_scores = []

        for im_id in im_ids:
            im = get_im_metadata_fn(im_id)
            rel_score = experimentmodel.relevance_score(im, caption, encode_caption_fn)

            rel_scores.append(rel_score)

        total_precision += experimentmodel.precision(rel_scores)
        total_ndcg += experimentmodel.ndcg(rel_scores)

    return ExperimentResult(
        mean_precision=total_precision / len(fixture.captions),
        mean_ndcg=total_ndcg / len(fixture.captions),
        mean_duration_sec=total_duration_sec / len(fixture.captions),
    )
