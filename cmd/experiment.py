import os

from dream import semsearch
from dream.semsearch.docstore import caption as docstorecaption
from dream import experiment


def _main() -> None:
    im_store_path = os.getenv("IM_STORE_PATH", "")
    _, _, semsearch_svc = semsearch.new_svc(im_store_path)

    caption_encoder = docstorecaption.CaptionFeatureExtractor()

    results = experiment.run_semsearch_experiments(semsearch_svc, caption_encoder.extract)

    for exp_name, res in results.items():
        print(exp_name, res.mean_duration_sec, res.mean_precision, res.mean_ndcg)


if __name__ == "__main__":
    _main()
