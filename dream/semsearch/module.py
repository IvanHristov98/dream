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
    captions_vt_store = vtstore.VocabularyTreeStore("captions_tree", "captions_node", "captions_train_job")
    captions_freq_store = vtstore.FrequencyStore("captions_tf", "captions_df", "captions_tree_doc_count")

    captions_vtree = vtservice.VocabularyTree(tx_store, captions_doc_store, captions_vt_store, captions_freq_store)

    mat_loader = imstore.MatrixLoader(imstore_ims_path)
    ims_doc_store = docstore.ImageStore(mat_loader)
    ims_vt_store = vtstore.VocabularyTreeStore("ims_tree", "ims_node", "ims_train_job")
    ims_freq_store = vtstore.FrequencyStore("ims_tf", "ims_df", "ims_tree_doc_count")

    ims_vtree = vtservice.VocabularyTree(tx_store, ims_doc_store, ims_vt_store, ims_freq_store)

    im_store = imstore.ImageStore(imstore_ims_path)
    semsearch_store = semsearchstore.Store(pool)

    sem_search_svc = semsearchservice.SemSearchService(captions_vtree, ims_vtree, im_store, semsearch_store)

    return (captions_vtree, ims_vtree, sem_search_svc)
