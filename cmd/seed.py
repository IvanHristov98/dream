import argparse
from pathlib import Path

import dream.voctree.store as revsearchstore
import dream.voctree.service as revsearchservice
import dream.voctree.imstore as revsearchimstore
from dream.voctree import featureextractor
from dream.controller import seed
from dream import dataset
from dream import logging as dreamlogging
from dream import pg as dreamdb


def main() -> None:
    dreamlogging.configure_logging()
    args = _parse_args()

    ds_iter = dataset.Coco2014Iterator(Path(args.coco2014_captions_path), Path(args.coco2014_ims_path))

    pool = dreamdb.new_pool()

    store = revsearchstore.Store(pool)
    im_store = revsearchimstore.FSImageStore(args.imstore_ims_path)
    feature_extractor = featureextractor.SiftExtractor()

    vtree = revsearchservice.VocabularyTree(store, im_store, feature_extractor)

    ctl = seed.SeedController(vtree, ds_iter)
    ctl.run()


def _parse_args() -> argparse.Namespace:
    # Only coco is supported as of now. This is due to change.
    parser = argparse.ArgumentParser()
    parser.add_argument("-coco2014-captions-path", help="Path to the COCO2014 captions file.", required=True, type=str)
    parser.add_argument("-coco2014-ims-path", help="Path to the COCO2014 images directory.", required=True, type=str)
    parser.add_argument("-imstore-ims-path", help="Path where to store the seeded images.", required=True, type=str)

    return parser.parse_args()


if __name__ == "__main__":
    main()
