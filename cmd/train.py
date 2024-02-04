import argparse

import dream.voctree.store as revsearchstore
import dream.voctree.imstore as revsearchimstore
from dream.voctree import featureextractor
import dream.voctree.service as revsearchservice
from dream import logging as dreamlogging
from dream import pg as dreamdb
from dream.controller import train


def main() -> None:
    dreamlogging.configure_logging()
    args = _parse_args()

    pool = dreamdb.new_pool()

    store = revsearchstore.Store(pool)
    im_store = revsearchimstore.FSImageStore(args.imstore_ims_path)
    feature_extractor = featureextractor.SiftExtractor()

    vtree = revsearchservice.VocabularyTree(store, im_store, feature_extractor)
    cfg = train.Config(sample_size=args.sample_size)

    controller = train.TrainController(cfg, vtree)
    controller.run()


def _parse_args() -> argparse.Namespace:
    # Only coco is supported as of now. This is due to change.
    parser = argparse.ArgumentParser()
    parser.add_argument("-imstore-ims-path", help="Path where to store the seeded images.", required=True, type=str)
    parser.add_argument("-sample-size", help="Approx. num of images in the training sample.", required=True, type=int)

    return parser.parse_args()


if __name__ == "__main__":
    main()
