import abc
import uuid
from typing import Callable, List, Optional, Dict, Tuple
import math

import sklearn.cluster as skcluster
import numpy as np

import dream.revsearch.model as rsmodel


class ErrNodeNotFound(Exception):
    """
    It is thrown whenever a node is not found.
    """


class DocStore(abc.ABC):
    def get_documents(self, tx: any, sample_size: int) -> List[rsmodel.Document]:
        raise NotImplementedError("")

    def get_feature_dim(self) -> int:
        raise NotImplementedError("")


class TreeStore(abc.ABC):
    def add_node(self, tx: any, node: rsmodel.Node) -> None:
        """
        add_node adds a new node in the store.
        """
        raise NotImplementedError("")

    def fetch_node_added_event(self, tx: any) -> Optional[rsmodel.NodeAdded]:
        """
        fetch_node_added_event fetches an arbitrary node added event.
        """
        raise NotImplementedError("")

    def remove_node_added_event(self, tx: any, event: rsmodel.NodeAdded) -> None:
        """
        remove_node_added_event deletes a node added event by its ID.
        """
        raise NotImplementedError("")

    def update_node(self, tx: any, node: rsmodel.Node) -> None:
        """
        update_node updates a given node.
        """
        raise NotImplementedError("")

    def get_root(self, tx: any) -> Optional[rsmodel.Node]:
        """
        get_root returns the root of the active vocabulary tree if any is found.
        """
        raise NotImplementedError("")

    def get_node(self, tx: any, id: uuid.UUID) -> Optional[rsmodel.Node]:
        """
        get_node returns a node by id.
        """
        raise NotImplementedError("")

    def is_training_in_progress(self, tx: any) -> bool:
        """
        is_training_in_progress checks whether all node_added events are consumed.
        """
        raise NotImplementedError("")

    def destroy_obsolete_nodes(self, tx: any) -> None:
        """
        destroy_obsolete_nodes destroys all nodes except the ones of the latest tree. Shouldn't be called while
        training is in progress so that we don't end up having partial vocabulary trees.

        It is a no-op when there are no nodes in the store.
        """
        raise NotImplementedError("")


class FrequencyStore(abc.ABC):
    def set_term(self, tx: any, term_id: uuid.UUID, frequencies: Dict[uuid.UUID, int]) -> None:
        """
        set_term stores how many times a term (a node) is found (reached) for each document (i.e. tf) and
        in how many unique documents it is found (i.e. df).
        """
        raise NotImplementedError("")

    def set_doc_counts(self, tx: any, tree_id: uuid.UUID, doc_counts: int) -> None:
        """
        set_doc_counts sets the total number of documents in a given tree.
        """
        raise NotImplementedError("")

    def get_df(self, tx: any, term_id: uuid.UUID) -> rsmodel.DocumentFrequency:
        """
        get_df returns how many unique documents a term (node) is found in.
        """
        raise NotImplementedError("")

    def get_doc_counts(self, tx: any, tree_id: uuid.UUID) -> int:
        """
        get_doc_counts gets the total number of documents in a given tree.
        """
        raise NotImplementedError("")

    def get_tfs(self, tx: any, term_id: uuid.UUID) -> Dict[uuid.UUID, int]:
        """
        get_tfs gets the number of times a term appears in each document.
        The returned dict only has values for documents it appears in. If a value for a document
        is missing it is assumed to be 0.
        """
        raise NotImplementedError("")


class Store(abc.ABC):
    def atomically(self, cb: Callable[[any], None]) -> None:
        raise NotImplementedError("")


class VocabularyTree:
    """
    VocabularyTree implements a vocabulary tree based index over a set documents.
    """

    _START_DEPTH = 0
    # Recommended values from the paper are 10 clusters and 6 levels.
    _NUM_CLUSTERS = 8
    _NUM_LEVELS = 4

    _store: Store
    _doc_store: DocStore
    _tree_store: TreeStore
    _freq_store: FrequencyStore
    _tf_idf_threshold: float

    def __init__(
        self,
        store: Store,
        doc_store: DocStore,
        tree_store: TreeStore,
        freq_store: FrequencyStore,
        # TODO: Fine tune tf_idf threshold.
        tf_idf_threshold: float = 0.05,
    ) -> None:
        self._store = store
        self._doc_store = doc_store
        self._tree_store = tree_store
        self._freq_store = freq_store
        self._tf_idf_threshold = tf_idf_threshold

    def start_training(self, sample_size: int) -> None:
        """
        train loads a subset of all documents in the db and adds a root node for a vocabulary tree.
        It might not use all documents to train the vocabulary tree due to memory constraints.
        If there are more than 2 trees it fails (one green and one blue).
        """

        def _cb(tx: any) -> None:
            feature_dim = self._doc_store.get_feature_dim()
            root_vec = np.zeros((feature_dim,), dtype=np.float32)
            root = rsmodel.Node(uuid.uuid4(), uuid.uuid4(), self._START_DEPTH, root_vec)

            docs = self._doc_store.get_documents(tx, sample_size)
            root.features = self._docs_to_features(docs)

            # TODO: What happens when more than two trees are added.
            self._tree_store.add_node(tx, root)

            # A term (node) is in a document if a feature of that document is in the node.
            # The term frequency for a given document is the number of features of a given
            # document that are in the node.
            frequencies = dict()

            for doc in docs:
                frequencies[doc.id] = len(doc.vectors)

            self._freq_store.set_term(tx, root.id, frequencies)

            self._freq_store.set_doc_counts(tx, root.tree_id, doc_counts=len(docs))

        self._store.atomically(_cb)

    def _docs_to_features(self, docs: List[rsmodel.Document]) -> List[rsmodel.Feature]:
        features = []

        for doc in docs:
            for vec in doc.vectors:
                feature = rsmodel.Feature(doc.id, vec)
                features.append(feature)

        return features

    def try_training(self) -> bool:
        """
        try_training fetches and consumes node added events one by one until none are left.
        Each event is consumed in a transaction of its own. A consumption consists of performing
        kmeans++ to cluster the features of a parent node and distribute them between all child nodes.

        Should be called periodically.

        Returns:
            bool: a boolean indicating whether there are more node added events to process.
        """
        worked = False

        def _cb(tx: any) -> None:
            nonlocal worked

            node_added = self._tree_store.fetch_node_added_event(tx)
            if node_added is None:
                return

            worked = True

            self._on_node_added(tx, node_added)
            self._tree_store.remove_node_added_event(tx, node_added)

        self._store.atomically(_cb)
        return worked

    def _on_node_added(self, tx: any, node_added: rsmodel.NodeAdded) -> None:
        if node_added.node.depth == self._NUM_LEVELS:
            # No more clustering of features is required.
            return

        parent = node_added.node
        children = self._get_children(parent)
        self._assign_children(parent, children)

        for child in children:
            self._tree_store.add_node(tx, child)

            doc_frequencies = self._get_document_frequencies()
            self._freq_store.set_term(tx, child.id, doc_frequencies)

        self._tree_store.update_node(tx, parent)

    def _get_children(self, parent: rsmodel.Node) -> List[rsmodel.Node]:
        vecs = self._get_features_vecs(parent.features)
        kmeans = skcluster.KMeans(init="k-means++", n_clusters=self._NUM_CLUSTERS, n_init="auto").fit(
            np.array(vecs),
        )

        if len(parent.features) != len(kmeans.labels_):
            # This error should not be encountered.
            raise RuntimeError("the number of features should be equal to the number of labels")

        children: Dict[int, rsmodel.Node] = dict()

        for i in range(len(kmeans.labels_)):
            label = kmeans.labels_[i]

            if label not in children:
                child_vec = kmeans.cluster_centers_[label]
                children[i] = rsmodel.Node(uuid.uuid4(), parent.tree_id, parent.depth + 1, child_vec)

            children[i].features.append(parent.features[i])

        return children

    def _get_features_vecs(self, features: List[rsmodel.Feature]) -> List[np.ndarray]:
        vecs = [None] * len(features)

        for i in range(len(features)):
            vecs[i] = features[i].vec

        return vecs

    def _assign_children(self, parent: rsmodel.Node, children: List[rsmodel.Node]) -> None:
        parent.features = set()

        for child in children.values():
            parent.children.add(child.id)

    def _get_document_frequencies(self, node: rsmodel.Node) -> Dict[uuid.UUID, int]:
        frequencies = dict()

        for feature in node.features:
            if feature.doc_id not in frequencies:
                frequencies[feature.doc_id] = 0

            frequencies[feature.doc_id] += 1

        return frequencies

    def try_replace_blue_tree(self) -> None:
        """
        try_replace_blue_tree deletes all obsolete tree nodes if no training is in progress.
        Should be called periodically.
        """

        def _cb(tx: any) -> None:
            if self._tree_store.is_training_in_progress(tx):
                return

            self._tree_store.destroy_obsolete_nodes(tx)

        self._store.atomically(_cb)

    def query(self, doc: rsmodel.Document, n: int) -> List[uuid.UUID]:
        """
        query finds the n most similar docs to the provided one and returns their IDs.
        It may return less than n docs at times.
        """
        doc_ids = []

        def _cb(tx: any) -> None:
            nonlocal doc_ids

            root = self._tree_store.get_root(tx)
            if root is None:
                raise ErrNodeNotFound("failed getting root of tree")

            query_tfs = self._find_term_frequencies(tx, doc, root)
            term_scores = self._find_term_scores(tx, doc, root, query_tfs)
            doc_ids = self._find_n_most_relevant_docs(doc, n, term_scores)

        self._store.atomically(_cb)
        return doc_ids

    def _find_term_frequencies(self, tx: any, doc: rsmodel.Document, root: rsmodel.Node) -> Dict[uuid.UUID, int]:
        query_tfs: Dict[uuid.UUID, int] = dict()
        nodes: Dict[uuid.UUID, rsmodel.Node] = dict()

        def _search_leaf(node: Optional[rsmodel.Node], vec: np.ndarray) -> None:
            nonlocal query_tfs
            nonlocal nodes
            nonlocal tx

            if node.id not in query_tfs:
                query_tfs[node.id] = 0

            query_tfs[node.id] += 1

            if len(node.children) == 0:
                # We've found a leaf.
                return

            dest = None
            min_dist = -1.0

            for child_id in node.children:
                # TODO: Caching of nodes for a single tx should probably be implemented in the store.
                if child_id in nodes:
                    child = nodes[child_id]
                else:
                    child = self._tree_store.get_node(tx, child_id)
                    if child is None:
                        raise ErrNodeNotFound(f"failed getting child node {child_id}")

                    nodes[child_id] = child

                dist = np.linalg.norm(vec - child.vec)
                if min_dist < 0 or min_dist > dist:
                    min_dist = dist
                    dest = child

            _search_leaf(dest, vec)

        for vec in doc.vectors:
            _search_leaf(root, vec)

        return query_tfs

    def _find_term_scores(
        self,
        tx: any,
        query_doc: rsmodel.Document,
        root: rsmodel.Node,
        query_tfs: Dict[uuid.UUID, int],
    ) -> Dict[uuid.UUID, Dict[uuid.UUID, float]]:
        term_scores: Dict[uuid.UUID, Dict[uuid.UUID, float]]

        docs_count = self._freq_store.get_doc_counts(tx, root.tree_id)

        for term_id, query_tf in query_tfs.items():
            df = self._freq_store.get_df(tx, term_id)

            # find score of query document
            norm_query_tf = query_tf / (query_tf + df.docs_count)
            idf = math.log((1 + docs_count) / (1 + df.unique_docs_count))

            query_tf_idf = norm_query_tf * idf
            if query_tf_idf < self._tf_idf_threshold:
                continue

            term_scores[term_id] = dict()
            term_scores[term_id][query_doc.id] = query_tf_idf

            tfs = self._freq_store.get_tfs(tx, term_id)
            for doc_id, tf in tfs.items():
                norm_tf = tf / (query_tf + df.docs_count)
                tf_idf = norm_tf * idf

                term_scores[term_id][doc_id] = tf_idf

        return term_scores

    def _find_n_most_relevant_docs(
        self, query_doc: rsmodel.Document, n: int, term_scores: Dict[uuid.UUID, Dict[uuid.UUID, float]]
    ) -> List[uuid.UUID]:
        doc_vecs : Dict[uuid.UUID, np.ndarray] = dict()
        term_ids = list(term_scores.keys())

        for i in range(term_ids):
            for doc_id, tf_idf in term_scores[term_ids[i]].items():
                if doc_id not in doc_vecs:
                    doc_vecs[doc_id] = np.zeros((len(term_ids),), dtype=np.float32)
                
                doc_vecs[doc_id][i] = tf_idf

        query_doc_vec = doc_vecs[query_doc.id]
        norm_query_doc_vec = query_doc_vec / np.linalg.norm(query_doc_vec, ord="1")
        doc_scores : List[Tuple[uuid.UUID, float]] = []

        for doc_id, doc_vec in doc_vecs.items():
            if doc_id == query_doc.id:
                continue

            # The authors of the vocabulary tree paper advise us to use 1st norm.
            # See https://people.eecs.berkeley.edu/~yang/courses/cs294-6/papers/nister_stewenius_cvpr2006.pdf.
            norm_doc_vec = doc_vec / np.linalg.norm(doc_vec, ord="1")

            doc_score = np.linalg.norm(norm_query_doc_vec - norm_doc_vec, ord="1")
            doc_scores.append((doc_id, doc_score))

        def _compare_fn(a, b):
            return a[1] - b[1]

        sorted(doc_scores, _compare_fn)
        return doc_scores[:min(n, len(doc_scores))]
