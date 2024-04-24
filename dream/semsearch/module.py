from typing import Tuple
from pathlib import Path

from dream import pg as dreamdb
from dream.semsearch import service as semsearchservice
from dream.semsearch import docstore
from dream.voctree import service as vtservice
from dream.voctree import store as vtstore
from dream.semsearch import imstore
from dream.semsearch import store as semsearchstore


def new_svc(
    imstore_ims_path: Path,
) -> Tuple[vtservice.VocabularyTree, vtservice.VocabularyTree, semsearchservice.SemSearchService]:
    pool = dreamdb.new_pool()

    tx_store = vtstore.TxStore(pool)

    captions_doc_store = docstore.CaptionStore()

    captions_tables = vtstore.TableConfig("captions")
    captions_vt_store = vtstore.VocabularyTreeStore(captions_tables)
    captions_freq_store = vtstore.FrequencyStore(captions_tables)
    captions_tree_reaper = vtstore.TreeReaper(captions_tables)

    captions_vtree = vtservice.VocabularyTree(
        tx_store, captions_doc_store, captions_vt_store, captions_freq_store, captions_tree_reaper
    )

    mat_loader = imstore.MatrixLoader(imstore_ims_path)
    ims_doc_store = docstore.ImageStore(mat_loader)

    ims_tables = vtstore.TableConfig("ims")
    ims_vt_store = vtstore.VocabularyTreeStore(ims_tables)
    ims_freq_store = vtstore.FrequencyStore(ims_tables)
    ims_tree_reaper = vtstore.TreeReaper(ims_tables)

    ims_vtree = vtservice.VocabularyTree(tx_store, ims_doc_store, ims_vt_store, ims_freq_store, ims_tree_reaper)

    im_store = imstore.ImageStore(imstore_ims_path)
    semsearch_store = semsearchstore.Store(pool)

    caption_feature_extractor = docstore.CaptionFeatureExtractor()
    im_feature_extractor = docstore.ImageFeatureExtractor()

    sem_search_svc = semsearchservice.SemSearchService(
        captions_vtree,
        ims_vtree,
        im_store,
        semsearch_store,
        caption_feature_extractor,
        im_feature_extractor,
    )

    return (captions_vtree, ims_vtree, sem_search_svc)
