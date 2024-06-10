import os

from dream import semsearch
from dream.semsearch.docstore import caption as docstorecaption
from dream import experiment
from dream import pg as dreampg


def _main() -> None:
    caption_encoder = docstorecaption.CaptionFeatureExtractor()

    pool = dreampg.new_pool()
    store = experiment.ImageMetadataStore(pool)

    ideal_relevances_per_caption = experiment.find_ideal_relevances(store, caption_encoder.extract)

    for caption, ideal_relevances in ideal_relevances_per_caption.items():
        print(caption, ideal_relevances)


if __name__ == "__main__":
    _main()
