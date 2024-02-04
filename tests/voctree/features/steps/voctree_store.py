from typing import Any
import uuid
import random

import numpy as np
from behave import *
from hamcrest import assert_that, is_, none, equal_to

import dream.voctree.store as store
import dream.voctree.model as vtmodel
import dream.pg as dreampg


_DIMS = 50


@given("a vocabulary tree store instance")
def step_impl(context):
    context.vt_store = store.VocabularyTreeStore("test_node", "test_train_job")


@given("blue tree nodes are created")
def step_impl(context):
    if "blue_tree" not in context:
        context.blue_tree = uuid.uuid4()

    if "nodes" not in context:
        context.nodes = dict()

    for row in context.table:
        vec = np.random.rand(_DIMS)
        context.nodes[row["node"]] = vtmodel.Node(uuid.uuid4(), context.blue_tree, int(row["depth"]), vec)


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
        vt_store: store.VocabularyTreeStore = context.vt_store
        node: vtmodel.Node = context.nodes[node_name]

        vt_store.update_node(tx, node)

    dreampg.with_tx(context.conn_pool, _cb)


@when("node {node_name} is fetched")
def step_impl(context, node_name):
    node: vtmodel.Node = context.nodes[node_name]

    def _cb(tx: Any) -> None:
        nonlocal context

        vt_store: store.VocabularyTreeStore = context.vt_store
        context.fetched_node = vt_store.get_node(tx, node.id)

    dreampg.with_tx(context.conn_pool, _cb)


@then("a null node is returned")
def step_impl(context):
    assert_that(context.fetched_node, is_(none()))


@then("node {node_name} is returned")
def step_impl(context, node_name):
    node: vtmodel.Node = context.nodes[node_name]

    assert_that(context.fetched_node, equal_to(node))
