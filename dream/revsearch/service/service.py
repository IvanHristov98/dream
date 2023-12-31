"""
Module service wraps the reverse search use cases according to clean architecture.
"""
import abc
import uuid
from typing import List, Callable, Dict

import sklearn.cluster as skcluster
import numpy as np
from tqdm import tqdm

import dream.revsearch.model as rsmodel
from dream import model


class TxStore(abc.ABC):
    def remove_all_nodes(self) -> None:
        raise NotImplementedError("")

    def store_node(self, _: rsmodel.Node) -> None:
        raise NotImplementedError("")

    def find_node(self, _: rsmodel.NodeID) -> rsmodel.Node:
        raise NotImplementedError("")

    def store_image_metadata(self, _: model.Image) -> None:
        raise NotImplementedError("")

    def find_image_metadata(self, _: model.ImageID) -> model.Image:
        raise NotImplementedError("")

    def load_training_images(self, sample_size: int) -> List[model.Image]:
        raise NotImplementedError("")


class Store(abc.ABC):
    def atomically(self, _: Callable[[TxStore], None]) -> None:
        raise NotImplementedError("")


class ImageStore(abc.ABC):
    def store_matrix(self, _: model.Image) -> None:
        raise NotImplementedError("")

    def load_matrix(self, _: model.Image) -> model.Image:
        raise NotImplementedError("")


class FeatureExtractor(abc.ABC):
    def features(self, _: model.Image) -> List[rsmodel.Feature]:
        raise NotImplementedError("")

    def dim(self) -> int:
        raise NotImplementedError("")


class VocabularyTree:
    """
    VocabularyTree implements use cases for reverse search of images.
    """

    # Value is taken from paper.
    # Recommended are 10 clusters and 6 levels.
    _NUM_CLUSTERS = 8
    _NUM_LEVELS = 4

    _store: Store
    _im_store: ImageStore
    _feature_extractor: FeatureExtractor

    def __init__(
        self,
        store: Store,
        im_store: ImageStore,
        feature_extractor: FeatureExtractor,
    ) -> None:
        self._store = store
        self._im_store = im_store
        self._feature_extractor = feature_extractor

    def seed_image(self, im: model.Image) -> None:
        self._im_store.store_matrix(im)

        # The code logic should always first search for image metadata in the db and then read the matrix.
        # Hence it is necessary to first write the matrix and then the metadata because they are operations
        # to different IO destinations.
        def _cb(tx_store: TxStore) -> None:
            tx_store.store_image_metadata(im)

        self._store.atomically(_cb)

    def train(self, sample_size: int) -> None:
        """
        train loads a subset of all images in the db, trains a vocabulary tree and stores it in the db.
        It might not associate all images with the tree.
        """

        def _cb(tx_store: TxStore) -> None:
            features = self._get_training_features(tx_store, sample_size)

            feature_dim = self._feature_extractor.dim()
            root_id = self._root_id()
            root = rsmodel.Node(id=root_id, vec=np.zeros((feature_dim,), dtype=np.float32), is_root=True)

            nodes: Dict[rsmodel.NodeID, rsmodel.Node] = dict()
            nodes[root.id] = root

            self._train_recursively(features, root, nodes)

            self._store_tree(tx_store, root, nodes)
            print(f"Added tree with root `{str(root.id)}` and removed all old nodes")

        self._store.atomically(_cb)

    def _root_id(self) -> rsmodel.NodeID:
        return rsmodel.NodeID(id=uuid.UUID(int=0))

    def _get_training_features(self, tx_store: TxStore, sample_size: int) -> List[rsmodel.Feature]:
        training_ims = tx_store.load_training_images(sample_size)
        features = []

        with tqdm(total=len(training_ims), desc="loading training ims") as pbar:
            for training_im in training_ims:
                loaded_im = self._im_store.load_matrix(training_im)

                im_features = self._feature_extractor.features(loaded_im)
                features += im_features
                pbar.update(1)

        return features

    def _train_recursively(
            self, 
            features: List[rsmodel.Feature], 
            node: rsmodel.Node, 
            nodes: Dict[rsmodel.NodeID, rsmodel.Node], 
            level: int = 1,
        ) -> None:
        if level == self._NUM_LEVELS:
            node.features = features
            return

        vecs = self._get_features_vecs(features)

        kmeans = skcluster.KMeans(init="k-means++", n_clusters=self._NUM_CLUSTERS, n_init="auto").fit(np.array(vecs))
        feature_groups = self._group_features(features, kmeans)

        for label in feature_groups.keys():
            child = rsmodel.Node(id=rsmodel.NodeID(uuid.uuid4()), vec=kmeans.cluster_centers_[label])
            nodes[child.id] = child

            node.add_child(child.id)

            self._train_recursively(feature_groups[label], child, nodes, level=level + 1)

    def _get_features_vecs(self, features: List[rsmodel.Feature]) -> List[np.ndarray]:
        vecs = [None] * len(features)

        for i in range(0, len(features)):
            vecs[i] = features[i].vec                

        return vecs

    def _group_features(
        self, features: List[rsmodel.Feature], kmeans: skcluster.KMeans
    ) -> Dict[int, List[rsmodel.Feature]]:
        groups: Dict[int, List[rsmodel.Feature]] = dict()

        for i in range(0, len(kmeans.labels_)):
            label = kmeans.labels_[i]

            if label not in groups:
                groups[label] = []

            groups[label].append(features[i])

        return groups

    def _store_ims(self, ims: List[model.Image]) -> None:
        for im in ims:
            self._im_store.store_matrix(im)

    def _store_tree(self, tx_store: TxStore, root: rsmodel.Node, nodes: Dict[rsmodel.NodeID, rsmodel.Node]) -> None:
        tx_store.remove_all_nodes()

        q: List[rsmodel.NodeID] = []
        q.append(root.id)

        while len(q) != 0:
            curr_id = q.pop(0)
            if curr_id not in nodes:
                # Shouldn't occur.
                raise KeyError("node id not found in nodes")

            curr = nodes[curr_id]
            q += list(curr.children)

            tx_store.store_node(curr)

    def _store_ims_metadata(self, tx_store: TxStore, ims: List[model.Image]) -> None:
        for im in ims:
            tx_store.store_image_metadata(im)

    # def find_similar(self, _: rsmodel.Feature) -> List[model.ImageID]:
    #     """
    #     find_similar finds images with features similar to the ones of the input image.
    #     """
    #     return None

    def find_image(self, _: model.ImageID) -> model.Image:
        """
        find_image finds an image with a given ID.
        """
        return None
