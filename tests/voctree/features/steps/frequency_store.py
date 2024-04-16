import uuid
from typing import Any, Dict

from behave import *
from hamcrest import assert_that, is_, none, equal_to, not_, has_key

from dream.voctree import store
import dream.pg as dreampg
import dream.voctree.model as vtmodel


@given("a frequency store instance")
def step_impl(context):
    tables = store.TableConfig("test")
    context.freq_store = store.FrequencyStore(tables)


@given("term {term_name} is created")
def step_impl(context, term_name):
    if "terms" not in context:
        context.terms = dict()

    context.terms[term_name] = uuid.uuid4()


@given("document {doc_name} is created")
def step_impl(context, doc_name):
    if "docs" not in context:
        context.docs = dict()

    context.docs[doc_name] = uuid.uuid4()


@given("frequencies for term {term_name} are inserted in {tree_name} tree")
def step_impl(context, term_name, tree_name):
    term_id = context.terms[term_name]
    frequencies: Dict[uuid.UUID, int] = dict()

    for row in context.table:
        doc_id = context.docs[row["doc"]]
        frequencies[doc_id] = int(row["frequency"])

    tree_id: uuid.UUID = context.trees[tree_name]

    def _cb(tx: Any) -> None:
        nonlocal context

        freq_store: store.FrequencyStore = context.freq_store
        freq_store.set_term(tx, term_id, frequencies, tree_id)

    dreampg.with_tx(context.conn_pool, _cb)


@given("documents count {docs_count} is inserted for {tree_name} tree")
def step_impl(context, docs_count, tree_name):
    tree_id: uuid.UUID = context.trees[tree_name]

    def _cb(tx: Any) -> None:
        nonlocal context

        freq_store: store.FrequencyStore = context.freq_store
        freq_store.set_doc_counts(tx, tree_id, int(docs_count))

    dreampg.with_tx(context.conn_pool, _cb)


@when("getting df of term {term_name}")
def step_impl(context, term_name):
    def _cb(tx: Any) -> None:
        nonlocal context

        freq_store: store.FrequencyStore = context.freq_store
        term_id = context.terms[term_name]

        context.fetched_df = freq_store.get_df(tx, term_id)

    dreampg.with_tx(context.conn_pool, _cb)


@when("getting tfs of term {term_name}")
def step_impl(context, term_name):
    def _cb(tx: Any) -> None:
        nonlocal context

        freq_store: store.FrequencyStore = context.freq_store
        term_id = context.terms[term_name]

        context.fetched_tfs = freq_store.get_tfs(tx, term_id)

    dreampg.with_tx(context.conn_pool, _cb)


@when("getting documents count of {tree_name} tree")
def step_impl(context, tree_name):
    tree_id: uuid.UUID = context.trees[tree_name]

    def _cb(tx: Any) -> None:
        nonlocal context

        freq_store: store.FrequencyStore = context.freq_store
        context.tree_docs_count = freq_store.get_doc_counts(tx, tree_id)

    dreampg.with_tx(context.conn_pool, _cb)


@then("no df is returned")
def step_impl(context):
    assert_that(context.fetched_df, is_(none()))


@then("fetched df has {unique_docs_count} unique docs")
def step_impl(context, unique_docs_count):
    fetched_df: vtmodel.DocumentFrequency = context.fetched_df
    assert_that(fetched_df.unique_docs_count, is_(equal_to(int(unique_docs_count))))


@then("it has a total tf of {total_tf}")
def step_impl(context, total_tf):
    fetched_df: vtmodel.DocumentFrequency = context.fetched_df
    assert_that(fetched_df.total_tf, is_(equal_to(int(total_tf))))


@then("it has tfs")
def step_impl(context):
    fetched_tfs: Dict[uuid.UUID, int] = context.fetched_tfs
    docs_count = 0

    for row in context.table:
        doc_id: uuid.UUID = context.docs[row["doc"]]
        expected_freq = int(row["frequency"])

        assert_that(fetched_tfs[doc_id], is_(equal_to(expected_freq)))

        docs_count += 1

    assert_that(len(fetched_tfs.values()), is_(equal_to(docs_count)))


@then("has no tf for doc {doc_name}")
def step_impl(context, doc_name):
    fetched_tfs: Dict[uuid.UUID, int] = context.fetched_tfs
    doc_id = context.docs[doc_name]

    assert_that(fetched_tfs, not_(has_key(doc_id)))


@then("it has no docs in tree")
def step_impl(context):
    assert_that(context.tree_docs_count, is_(none()))


@then("it has {docs_count} docs in tree")
def step_impl(context, docs_count):
    assert_that(context.tree_docs_count, is_(equal_to(int(docs_count))))
