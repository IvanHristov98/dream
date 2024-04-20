from typing import List
import logging
import uuid

import psycopg

from dream import model


_BATCH_SIZE = 100


def sample_im_metadata(
    pg_tx: psycopg.Cursor, sample_size: int, tree_id: uuid.UUID, doc_store: str
) -> List[model.Image]:
    im_count = _get_unsampled_im_count(pg_tx, tree_id, doc_store)

    if im_count < sample_size:
        logging.warning(
            "sample_size (%d) is greater than training dataset size (%d)",
            sample_size,
            im_count,
        )
        sample_size = im_count

    if sample_size < 0:
        logging.warning("sample_size must be > 0, setting it to ~1%")
        sample_size = im_count / 100 + 1

    ims = _load_im_metadata(pg_tx, sample_size, im_count, tree_id, doc_store)
    _mark_sampled(pg_tx, ims, tree_id, doc_store)
    return ims


def _get_unsampled_im_count(pg_tx: psycopg.Cursor, tree_id: uuid.UUID, doc_store: str) -> int:
    pg_tx.execute("""SELECT COUNT(*) FROM image_metadata im
        WHERE im.id NOT IN (
            SELECT im_id FROM image_metadata_sample_log imsl
            WHERE imsl.im_id=im.id AND imsl.tree_id=%s AND imsl.doc_store=%s
        )""", (tree_id, doc_store))

    row = pg_tx.fetchone()
    return row[0]


def _load_im_metadata(
    pg_tx: psycopg.Cursor, sample_size: int, im_count: int, tree_id: uuid.UUID, doc_store: str
) -> List[model.Image]:
    ratio = min(sample_size / im_count, 1)
    query = f"""SELECT id, captions, dataset FROM image_metadata im
        WHERE im.id NOT IN (
            SELECT im_id FROM image_metadata_sample_log imsl
            WHERE imsl.im_id=im.id AND imsl.tree_id=%s AND imsl.doc_store=%s
        ) AND random() < {ratio};"""

    pg_tx.execute(query, (tree_id, doc_store))
    ims = []

    for row in pg_tx.fetchall():
        im = model.Image().with_id(row[0]).with_dataset(row[2]).with_captions(row[1])
        ims.append(im)

    return ims


def _mark_sampled(pg_tx: psycopg.Cursor, ims: List[model.Image], tree_id: uuid.UUID, doc_store: str) -> None:
    for i in range(0, len(ims), _BATCH_SIZE):
        ims_slice = ims[i : min(i + _BATCH_SIZE, len(ims))]

        _mark_sampled_batch(pg_tx, ims_slice, tree_id, doc_store)


def _mark_sampled_batch(pg_tx: psycopg.Cursor, ims: List[model.Image], tree_id: uuid.UUID, doc_store: str) -> None:
    query_parts = ["INSERT INTO image_metadata_sample_log (im_id, tree_id, doc_store) VALUES"]
    vals = []

    for i, im in enumerate(ims):
        if i == len(ims) - 1:
            query_parts.append(f"(%s, %s, '{doc_store}');")
        else:
            query_parts.append(f"(%s, %s, '{doc_store}'),")

        vals += [im.id.hex, tree_id.hex]

    query = " ".join(query_parts)
    pg_tx.execute(query, tuple(vals))
