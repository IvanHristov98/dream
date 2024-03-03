from typing import Any, List
import uuid
import random
import threading

import numpy as np
from behave import *
from hamcrest import assert_that, is_, none, equal_to, is_in, calling, raises, has_items

from dream.voctree import store
import dream.voctree.model as vtmodel
import dream.pg as dreampg


_DIMS = 50


@given("a vocabulary tree store instance")
def step_impl(context):
    context.vt_store = store.VocabularyTreeStore("test_tree", "test_node", "test_train_job")


@given("{tree_name} tree is created")
def step_impl(context, tree_name):
    if "trees" not in context:
        context.trees = dict()

    context.trees[tree_name] = uuid.uuid4()


@given("{tree_name} tree nodes are created")
def step_impl(context, tree_name):
    tree_id = context.trees[tree_name]

    if "nodes" not in context:
        context.nodes = dict()

    for row in context.table:
        vec = np.random.rand(_DIMS)
        context.nodes[row["node"]] = vtmodel.Node(uuid.uuid4(), tree_id, int(row["depth"]), vec)


@given("{tree_name} tree is inserted")
def step_impl(context, tree_name):
    def _cb(tx: Any) -> None:
        nonlocal context

        tree_id = context.trees[tree_name]
        vt_store: store.VocabularyTreeStore = context.vt_store
        vt_store.add_tree(tx, tree_id)

    dreampg.with_tx(context.conn_pool, _cb)


@given("node relations are added")
def step_impl(context):
    for row in context.table:
        child: vtmodel.Node = context.nodes[row["child"]]
        parent: vtmodel.Node = context.nodes[row["parent"]]

        parent.children.add(child.id)
        context.nodes[row["parent"]] = parent


@given("nodes are leafs")
def step_impl(context):
    for row in context.table:
        node: vtmodel.Node = context.nodes[row["node"]]

        feature_count = random.randint(1, 10)
        for _ in range(feature_count):
            vec = np.random.rand(_DIMS)
            feature = vtmodel.Feature(doc_id=uuid.uuid4(), vec=vec)

            node.features.append(feature)

        context.nodes[row["node"]] = node


@given("node {node_name} is inserted")
def step_impl(context, node_name):
    def _cb(tx: Any) -> None:
        nonlocal context

        vt_store: store.VocabularyTreeStore = context.vt_store
        node: vtmodel.Node = context.nodes[node_name]

        vt_store.add_node(tx, node)

    dreampg.with_tx(context.conn_pool, _cb)


@given("{node_name} is not leaf anymore")
def step_impl(context, node_name):
    node: vtmodel.Node = context.nodes[node_name]
    node.children = set()

    context.nodes[node_name] = node


@given("{node_name} is updated")
def step_impl(context, node_name):
    def _cb(tx: Any) -> None:
        nonlocal context

        vt_store: store.VocabularyTreeStore = context.vt_store
        node: vtmodel.Node = context.nodes[node_name]

        vt_store.update_node(tx, node)

    dreampg.with_tx(context.conn_pool, _cb)


@given("a train job for node {node_name} is created")
def step_impl(context, node_name):
    node: vtmodel.Node = context.nodes[node_name]

    if "train_jobs" not in context:
        context.train_jobs = dict()

    context.train_jobs[node_name] = vtmodel.TrainJob(uuid.uuid4(), node)


@given("a train job for node {node_name} is pushed")
def step_impl(context, node_name):
    def _cb(tx: Any) -> None:
        nonlocal context

        vt_store: store.VocabularyTreeStore = context.vt_store
        train_job: vtmodel.TrainJob = context.train_jobs[node_name]

        vt_store.push_train_job(tx, train_job)

    dreampg.with_tx(context.conn_pool, _cb)


@given("train job for node {node_name} is popped")
def step_impl(context, node_name):
    def _cb(tx: Any) -> None:
        nonlocal context

        vt_store: store.VocabularyTreeStore = context.vt_store
        train_job: vtmodel.TrainJob = context.train_jobs[node_name]

        vt_store.pop_train_job(tx, train_job.id)

    dreampg.with_tx(context.conn_pool, _cb)


@given("cleanup is performed")
def step_impl(context):
    def _cb(tx: Any) -> None:
        nonlocal context

        vt_store: store.VocabularyTreeStore = context.vt_store
        vt_store.cleanup(tx)

    dreampg.with_tx(context.conn_pool, _cb)


@when("node {node_name} is fetched")
def step_impl(context, node_name):
    node: vtmodel.Node = context.nodes[node_name]

    def _cb(tx: Any) -> None:
        nonlocal context

        vt_store: store.VocabularyTreeStore = context.vt_store
        context.fetched_node = vt_store.get_node(tx, node.id)

    dreampg.with_tx(context.conn_pool, _cb)


@when("root is fetched")
def step_impl(context):
    def _cb(tx: Any) -> None:
        nonlocal context

        vt_store: store.VocabularyTreeStore = context.vt_store
        context.fetched_node = vt_store.get_root(tx)

    dreampg.with_tx(context.conn_pool, _cb)


@when("train job is fetched")
def step_impl(context):
    def _cb(tx: Any) -> None:
        nonlocal context

        vt_store: store.VocabularyTreeStore = context.vt_store
        context.fetched_train_job = vt_store.fetch_train_job(tx)

    dreampg.with_tx(context.conn_pool, _cb)


@when("{n} train jobs are fetched in parallel txs")
def step_impl(context, n):
    barrier = threading.Barrier(int(n))
    fetched_jobs = []

    def _train_fn():
        def _cb(tx: Any) -> None:
            nonlocal context
            nonlocal barrier
            nonlocal fetched_jobs

            vt_store: store.VocabularyTreeStore = context.vt_store
            fetched_jobs.append(vt_store.fetch_train_job(tx))

            barrier.wait()

        dreampg.with_tx(context.conn_pool, _cb)

    workers: List[threading.Thread] = []

    for _ in range(int(n)):
        t = threading.Thread(target=_train_fn)
        t.start()

        workers.append(t)

    for worker in workers:
        worker.join()

    context.fetched_train_jobs = list(filter(lambda x: x is not None, fetched_jobs))


@then("a null node is returned")
def step_impl(context):
    assert_that(context.fetched_node, is_(none()))


@then("node {node_name} is returned")
def step_impl(context, node_name):
    node: vtmodel.Node = context.nodes[node_name]

    assert_that(context.fetched_node, is_(equal_to(node)))


@then("one of nodes is returned")
def step_impl(context):
    nodes: List[vtmodel.Node] = []

    for row in context.table:
        node = context.nodes[row["node"]]
        nodes.append(node)

    assert_that(context.fetched_node, is_in(nodes))


@then("a null train job is returned")
def step_impl(context):
    assert_that(context.fetched_train_job, is_(none()))


@then("a train job for node {node_name} is returned")
def step_impl(context, node_name):
    train_job: vtmodel.TrainJob = context.train_jobs[node_name]

    assert_that(context.fetched_train_job, is_(equal_to(train_job)))


@then("inserting {tree_name} tree raises an exception")
def step_impl(context, tree_name):
    tree_id: uuid.UUID = context.trees[tree_name]

    def _cb(tx: Any) -> None:
        nonlocal context

        vt_store: store.VocabularyTreeStore = context.vt_store
        vt_store.add_tree(tx, tree_id)

    assert_that(calling(dreampg.with_tx).with_args(context.conn_pool, _cb), raises(Exception))


@then("train jobs are returned")
def step_impl(context):
    train_jobs: List[vtmodel.Node] = []

    for row in context.table:
        train_job = context.train_jobs[row["node"]]
        train_jobs.append(train_job)

    assert_that(context.fetched_train_jobs, has_items(*train_jobs))
