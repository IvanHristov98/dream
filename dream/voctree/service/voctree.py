import abc
import uuid
from typing import Callable, List, Optional, Dict, Tuple, Set
import math
import logging

import sklearn.cluster as skcluster
import numpy as np

import dream.voctree.model as vtmodel
import dream.voctree.api as vtapi


class VocabularyTreeStore(abc.ABC):
    def add_tree(self, tx: any, tree_id: uuid.UUID) -> None:
        """
        add_tree adds a new tree. There can be at most 2 trees - a blue and a green one.
        """
        raise NotImplementedError("")

    def add_node(self, tx: any, node: vtmodel.Node) -> None:
        """
        add_node adds a new node.
        """
        raise NotImplementedError("")

    def push_train_job(self, tx: any, job: vtmodel.TrainJob) -> None:
        """
        push_train_job pushes a new train job.
        """
        raise NotImplementedError("")

    def fetch_train_job(self, tx: any) -> Optional[vtmodel.TrainJob]:
        """
        fetch_train_job fetches an arbitrary training job.
        """
        raise NotImplementedError("")

    def pop_train_job(self, tx: any, job_id: uuid.UUID) -> None:
        """
        pop_train_job deletes a training job by its ID.
        """
        raise NotImplementedError("")

    def update_node(self, tx: any, node: vtmodel.Node) -> None:
        """
        update_node updates a given node.
        """
        raise NotImplementedError("")

    def get_root(self, tx: any) -> Optional[vtmodel.Node]:
        """
        get_root returns the root of the active vocabulary tree if any is found.
        """
        raise NotImplementedError("")

    def get_node(self, tx: any, node_id: uuid.UUID) -> Optional[vtmodel.Node]:
        """
        get_node returns a node by id.
        """
        raise NotImplementedError("")


class FrequencyStore(abc.ABC):
    def set_term(self, tx: any, term_id: uuid.UUID, frequencies: Dict[uuid.UUID, int], tree_id: uuid.UUID) -> None:
        """
        set_term stores how many times a term (a node) is found (reached) for each document (i.e. tf) and
        in how many unique documents it is found (i.e. df).
        """
        raise NotImplementedError("")

    def set_doc_counts(self, tx: any, tree_id: uuid.UUID, docs_count: int) -> None:
        """
        set_doc_counts sets the total number of documents in a given tree.
        """
        raise NotImplementedError("")

    def get_df(self, tx: any, term_id: uuid.UUID) -> Optional[vtmodel.DocumentFrequency]:
        """
        get_df returns how many unique documents a term (node) is found in.
        """
        raise NotImplementedError("")

    def get_doc_counts(self, tx: any, tree_id: uuid.UUID) -> Optional[int]:
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


class TreeReaper(abc.ABC):
    def reap_deprecated_trees(self, tx: any) -> None:
        """
        cleanup removes all information stored in relation to deprecated trees, including the trees themselves.
        The goal is to reduce disk usage.
        """
        raise NotImplementedError("")


class TxStore(abc.ABC):
    def in_tx(self, cb: Callable[[any], None]) -> None:
        raise NotImplementedError("")


class VocabularyTree(vtapi.VocabularyTree):
    """
    VocabularyTree implements a vocabulary tree based index over a set documents.
    """

    _START_DEPTH = 0
    # Recommended values from the paper are 10 clusters and 6 levels.
    _NUM_CLUSTERS = 10
    _NUM_LEVELS = 6
    _MIN_SAMPLES = 100
    # TODO: Fine tune tf_idf threshold.
    _TF_IDF_THRESHOLD = 0.05
    _POPULATE_BATCH_SIZE = 100

    _tx_store: TxStore
    _doc_store: vtapi.DocStore
    _tree_store: VocabularyTreeStore
    _freq_store: FrequencyStore
    _tree_reaper: TreeReaper

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        tx_store: TxStore,
        doc_store: vtapi.DocStore,
        tree_store: VocabularyTreeStore,
        freq_store: FrequencyStore,
        tree_reaper: TreeReaper,
    ) -> None:
        self._tx_store = tx_store
        self._doc_store = doc_store
        self._tree_store = tree_store
        self._freq_store = freq_store
        self._tree_reaper = tree_reaper

    def start_training(self, sample_size: int) -> None:
        """
        start_training loads a subset of all documents in the db and adds a root node for a vocabulary tree.

        It might not use all documents to train the vocabulary tree due to memory constraints.
        The rest of the documents would be populated into the tree slowly over time after initial training.

        If there are more than 2 trees it fails (one green and one blue).
        """

        def _cb(tx: any) -> None:
            tree_id = uuid.uuid4()
            self._tree_store.add_tree(tx, tree_id)

            feature_dim = self._doc_store.get_feature_dim()
            root_vec = np.zeros((feature_dim,), dtype=np.float32)
            root = vtmodel.Node(uuid.uuid4(), tree_id, self._START_DEPTH, root_vec)

            docs = self._doc_store.sample_next_documents(tx, sample_size, root.tree_id)
            logging.info("fetched %d documents", len(docs))
            root.features = self._docs_to_features(docs)

            # TODO: What happens when more than two trees are added -> Raise exception.
            self._tree_store.add_node(tx, root)
            logging.info("pushed root node %s", root.id)
            self._tree_store.push_train_job(tx, vtmodel.TrainJob(job_id=uuid.uuid4(), added_node=root))
            logging.info("added train job for root node %s", root.id)

            # A term (node) is in a document if a feature of that document is in the node.
            # The term frequency for a given document is the number of features of a given
            # document that are in the node.
            frequencies = {}

            for doc in docs:
                frequencies[doc.id] = len(doc.vectors)

            self._freq_store.set_term(tx, root.id, frequencies, root.tree_id)

            self._freq_store.set_doc_counts(tx, root.tree_id, docs_count=len(docs))

            logging.info("stored frequency info for root node %s", root.id)

        self._tx_store.in_tx(_cb)

    def _docs_to_features(self, docs: List[vtapi.Document]) -> List[vtmodel.Feature]:
        features = []

        for doc in docs:
            for vec in doc.vectors:
                feature = vtmodel.Feature(doc.id, vec)
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

            train_job = self._tree_store.fetch_train_job(tx)
            if train_job is None:
                return

            worked = True

            self._on_train_job(tx, train_job)
            self._tree_store.pop_train_job(tx, train_job.id)

        self._tx_store.in_tx(_cb)
        return worked

    def _on_train_job(self, tx: any, train_job: vtmodel.TrainJob) -> None:
        if train_job.added_node.depth == self._NUM_LEVELS:
            # No more clustering of features is required.
            return

        parent = train_job.added_node
        children = self._get_children(parent)
        if children is None:
            return

        self._assign_children(parent, children)

        for child in children.values():
            self._tree_store.add_node(tx, child)
            self._tree_store.push_train_job(tx, vtmodel.TrainJob(job_id=uuid.uuid4(), added_node=child))

            doc_frequencies = self._get_document_frequencies(child)
            self._freq_store.set_term(tx, child.id, doc_frequencies, train_job.added_node.tree_id)

        self._tree_store.update_node(tx, parent)

    def _get_children(self, parent: vtmodel.Node) -> Optional[Dict[int, vtmodel.Node]]:
        vecs = self._get_features_vecs(parent.features)
        if len(vecs) < self._MIN_SAMPLES:
            return None

        kmeans = skcluster.KMeans(init="k-means++", n_clusters=self._NUM_CLUSTERS, n_init="auto").fit(
            np.array(vecs),
        )

        if len(parent.features) != len(kmeans.labels_):
            # This error should not be encountered.
            raise RuntimeError("the number of features should be equal to the number of labels")

        children: Dict[int, vtmodel.Node] = {}

        for i, label in enumerate(kmeans.labels_):
            if label not in children:
                child_vec = kmeans.cluster_centers_[label]
                children[label] = vtmodel.Node(uuid.uuid4(), parent.tree_id, parent.depth + 1, child_vec)

            children[label].features.append(parent.features[i])

        return children

    def _get_features_vecs(self, features: List[vtmodel.Feature]) -> List[np.ndarray]:
        vecs = [None] * len(features)

        for i, feature in enumerate(features):
            vecs[i] = feature.vec

        return vecs

    def _assign_children(self, parent: vtmodel.Node, children: List[vtmodel.Node]) -> None:
        parent.features = set()

        for child in children.values():
            parent.children.add(child.id)

    def _get_document_frequencies(self, node: vtmodel.Node) -> Dict[uuid.UUID, int]:
        frequencies = {}

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
            self._tree_reaper.reap_deprecated_trees(tx)

        self._tx_store.in_tx(_cb)

    def try_populate_tree(self) -> bool:
        """
        try_populate_tree finds the root of the currently active tree and adds documents to it.
        If useful Voronoi cells are split.
        """
        worked = False

        def _cb(tx: any) -> None:
            nonlocal worked

            root = self._tree_store.get_root(tx)
            if root is None:
                return

            docs = self._doc_store.sample_next_documents(tx, self._POPULATE_BATCH_SIZE, root.tree_id)
            if len(docs) == 0:
                return

            nodes_cache: Dict[uuid.UUID, vtmodel.Node] = {}
            visited_leafs: Set[uuid.UUID] = set()

            for doc in docs:
                doc_leafs = self._try_populate_doc(tx, nodes_cache, root, doc)
                visited_leafs = visited_leafs.union(doc_leafs)

            for visited_leaf in visited_leafs:
                self._update_visited_leaf(tx, nodes_cache[visited_leaf])

            worked = True

        self._tx_store.in_tx(_cb)
        return worked

    def _try_populate_doc(
        self,
        tx: any,
        nodes_cache: Dict[uuid.UUID, vtmodel.Node],
        root: vtmodel.Node,
        doc: vtapi.Document,
    ) -> Set[uuid.UUID]:
        visited_leafs: Set[uuid.UUID] = set()

        def _on_visit_fn(node: vtmodel.Node, vec: np.ndarray) -> None:
            nonlocal visited_leafs

            if not self._is_leaf(node):
                return

            node.features.append(vtmodel.Feature(doc.id, vec))
            visited_leafs.add(node.id)

        for vec in doc.vectors:
            self._search_leaf(tx, nodes_cache, root, vec, _on_visit_fn)

        return visited_leafs

    def _update_visited_leaf(
        self,
        tx: any,
        visited_leaf: vtmodel.Node,
    ) -> None:
        children = self._get_children(visited_leaf)
        if children is None:
            # Only save the added features.
            self._tree_store.update_node(tx, visited_leaf)
            return

        # Case when the leaf has become too big and needs to be split.
        self._split_leaf(tx, visited_leaf, children)

    def _split_leaf(self, tx: any, visited_leaf: vtmodel.Node, children: Dict[int, vtmodel.Node]) -> None:
        self._assign_children(visited_leaf, children)

        # TODO: Handle corner case when one of the children could also be broken down into children.
        # Currently it is assumed that it has a negligible effect.
        for child in children.values():
            self._tree_store.add_node(tx, child)

            doc_frequencies = self._get_document_frequencies(child)
            self._freq_store.set_term(tx, child.id, doc_frequencies, visited_leaf.tree_id)

        self._tree_store.update_node(tx, visited_leaf)

    def query(self, doc: vtapi.Document, n: int) -> List[uuid.UUID]:
        """
        query finds the n most similar docs to the provided one and returns their IDs.
        It may return less than n docs at times.
        """
        doc_ids = []

        def _cb(tx: any) -> None:
            nonlocal doc_ids

            root = self._tree_store.get_root(tx)
            if root is None:
                raise vtapi.ErrNodeNotFound("getting root of tree")

            query_tfs = self._find_term_frequencies(tx, doc, root)
            term_scores = self._find_term_scores(tx, doc, root, query_tfs)
            doc_ids = self._find_n_most_relevant_docs(doc, n, term_scores)

        self._tx_store.in_tx(_cb)
        return doc_ids

    def _find_term_frequencies(self, tx: any, doc: vtapi.Document, root: vtmodel.Node) -> Dict[uuid.UUID, int]:
        query_tfs: Dict[uuid.UUID, int] = {}
        nodes_cache: Dict[uuid.UUID, vtmodel.Node] = {}

        def _on_visit_fn(node: vtmodel.Node, _: np.ndarray) -> None:
            nonlocal query_tfs

            if node.id not in query_tfs:
                query_tfs[node.id] = 0

            query_tfs[node.id] += 1

        for vec in doc.vectors:
            self._search_leaf(tx, nodes_cache, root, vec, _on_visit_fn)

        return query_tfs

    # pylint: disable-next=too-many-arguments
    def _search_leaf(
        self,
        tx: any,
        nodes_cache: Dict[uuid.UUID, vtmodel.Node],
        node: Optional[vtmodel.Node],
        vec: np.ndarray,
        on_visit_fn: Callable[[vtmodel.Node, np.ndarray], None],
    ) -> None:
        on_visit_fn(node, vec)

        if self._is_leaf(node):
            # We've found a leaf.
            return

        dest = None
        min_dist = -1.0

        for child_id in node.children:
            if child_id in nodes_cache:
                child = nodes_cache[child_id]
            else:
                child = self._tree_store.get_node(tx, child_id)
                if child is None:
                    raise vtapi.ErrNodeNotFound(f"failed getting child node {child_id}")

                nodes_cache[child_id] = child

            dist = np.linalg.norm(vec - child.vec)
            if min_dist < 0 or min_dist > dist:
                min_dist = dist
                dest = child

        self._search_leaf(tx, nodes_cache, dest, vec, on_visit_fn)

    def _is_leaf(self, node: vtmodel.Node) -> bool:
        return len(node.children) == 0

    def _find_term_scores(
        self,
        tx: any,
        query_doc: vtapi.Document,
        root: vtmodel.Node,
        query_tfs: Dict[uuid.UUID, int],
    ) -> Dict[uuid.UUID, Dict[uuid.UUID, float]]:
        term_scores: Dict[uuid.UUID, Dict[uuid.UUID, float]] = {}

        docs_count = self._freq_store.get_doc_counts(tx, root.tree_id)

        if docs_count is None:
            raise RuntimeError("there should always be a total documents count")

        for term_id, query_tf in query_tfs.items():
            df = self._freq_store.get_df(tx, term_id)
            if df is None:
                raise RuntimeError("df should alway be there")

            # find score of query document
            norm_query_tf = query_tf / (query_tf + df.total_tf)
            idf = math.log((1 + docs_count) / (1 + df.unique_docs_count))

            query_tf_idf = norm_query_tf * idf
            if query_tf_idf < self._TF_IDF_THRESHOLD:
                continue

            term_scores[term_id] = {}
            term_scores[term_id][query_doc.id] = query_tf_idf

            tfs = self._freq_store.get_tfs(tx, term_id)
            for doc_id, tf in tfs.items():
                norm_tf = tf / (query_tf + df.total_tf)
                tf_idf = norm_tf * idf

                term_scores[term_id][doc_id] = tf_idf

        return term_scores

    def _find_n_most_relevant_docs(
        self, query_doc: vtapi.Document, n: int, term_scores: Dict[uuid.UUID, Dict[uuid.UUID, float]]
    ) -> List[uuid.UUID]:
        doc_vecs: Dict[uuid.UUID, np.ndarray] = {}
        term_ids = list(term_scores.keys())

        for i in range(len(term_ids)):
            for doc_id, tf_idf in term_scores[term_ids[i]].items():
                if doc_id not in doc_vecs:
                    doc_vecs[doc_id] = np.zeros((len(term_ids),), dtype=np.float32)

                doc_vecs[doc_id][i] = tf_idf

        query_doc_vec = doc_vecs[query_doc.id]
        norm_query_doc_vec = query_doc_vec / np.linalg.norm(query_doc_vec, ord=1)
        doc_scores: List[Tuple[uuid.UUID, float]] = []

        for doc_id, doc_vec in doc_vecs.items():
            if doc_id == query_doc.id:
                continue

            # The authors of the vocabulary tree paper advise us to use 1st norm.
            # See https://people.eecs.berkeley.edu/~yang/courses/cs294-6/papers/nister_stewenius_cvpr2006.pdf.
            norm_doc_vec = doc_vec / np.linalg.norm(doc_vec, ord=1)

            doc_score = np.linalg.norm(norm_query_doc_vec - norm_doc_vec, ord=1)
            doc_scores.append((doc_id, doc_score))

        def _compare_fn(a):
            return a[1]

        doc_scores = sorted(doc_scores, key=_compare_fn)

        doc_scores = doc_scores[: min(n, len(doc_scores))]

        doc_ids = []
        for doc_score in doc_scores:
            doc_ids.append(doc_score[0])

        return doc_ids
