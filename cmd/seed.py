import logging
import sys
import argparse
import os
from pathlib import Path

import psycopg_pool

import dream.revsearch.store as revsearchstore
import dream.revsearch.service as revsearchservice
import dream.revsearch.imstore as revsearchimstore
from dream.revsearch import featureextractor
from dream.controller import seed
from dream import dataset


def main() -> None:
    _configure_logging()
    args = _parse_args()

    ds_iter = dataset.Coco2014Iterator(Path(args.coco2014_captions_path), Path(args.coco2014_ims_path))

    conn_info = os.getenv("DB_CONN_INFO")  # e.g. `host=localhost port=5432 dbname=dream user=dream password=pass`
    pool = psycopg_pool.ConnectionPool(conn_info)

    store = revsearchstore.Store(pool)
    im_store = revsearchimstore.FSImageStore(args.imstore_ims_path)
    feature_extractor = featureextractor.SiftExtractor()

    vtree = revsearchservice.VocabularyTree(store, im_store, feature_extractor)

    ctl = seed.SeedController(vtree, ds_iter)
    ctl.run()


def _configure_logging() -> None:
    fmt = "[%(asctime)s][%(levelname)s] %(message)s"
    logging.basicConfig(stream=sys.stdout, format=fmt, level=logging.INFO, datefmt="%H:%M:%S")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-coco2014-captions-path", help="Path to the COCO2014 captions file.", required=True, type=str)
    parser.add_argument("-coco2014-ims-path", help="Path to the COCO2014 images directory.", required=True, type=str)
    parser.add_argument("-imstore-ims-path", help="Path where to store the seeded images.", required=True, type=str)

    return parser.parse_args()


if __name__ == "__main__":
    main()
