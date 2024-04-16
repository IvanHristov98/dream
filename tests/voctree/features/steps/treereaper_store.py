from typing import Any

from behave import *

from dream.voctree import store
import dream.pg as dreampg


@given("a tree reaper instance")
def step_impl(context):
    tables = store.TableConfig("test")
    context.tree_reaper = store.TreeReaper(tables)


@given("cleanup is performed")
def step_impl(context):
    def _cb(tx: Any) -> None:
        nonlocal context

        tree_reaper: store.TreeReaper = context.tree_reaper
        tree_reaper.reap_deprecated_trees(tx)

    dreampg.with_tx(context.conn_pool, _cb)
