-- 
-- depends:

CREATE TABLE IF NOT EXISTS node (
	id UUID  PRIMARY KEY,
	tree_id  UUID NOT NULL,
	depth    INT NOT NULL,
	vec      BYTEA NOT NULL,
	im_count INT NOT NULL,
	children JSONB NOT NULL,
	features JSONB NOT NULL,
);

CREATE INDEX IF NOT EXISTS node_tree_id_depth_idx ON node (tree_id, depth);

CREATE TABLE IF NOT EXISTS node_added_event (
	id         UUID PRIMARY KEY,
	node_id    UUID NOT NULL,
	created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

	CONSTRAINT fk_node_id FOREIGN KEY (node) REFERENCES node(id)
);

CREATE TABLE IF NOT EXISTS caption_tf (
	-- term_id is a node in the tree.
	term_id UUID NOT NULL,
	-- doc_id is a document in the vocabulary tree.
	doc_id UUID NOT NULL,
	-- frequency is the number of times the term is found for the document,
	-- i.e. the number of descriptor vectors of the document that the node contains.
	frequency INT NOT NULL,
	PRIMARY KEY (term_id, doc_id)
);

CREATE TABLE IF NOT EXISTS caption_df (
	term_id           UUID PRIMARY KEY,
	unique_docs_count INT NOT NULL,
	docs_count        INT NOT NULL,
);

CREATE TABLE IF NOT EXISTS doc_counts (
	tree_id UUID NOT NULL,
	-- doc_count is the number of documents in a given vocabulary tree.
	doc_count INT NOT NULL,
);

CREATE TABLE IF NOT EXISTS image_metadata (
	id      UUID PRIMARY KEY,
	labels  JSONB NOT NULL,
	dataset TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS image_metadata_dataset_idx ON image_metadata (dataset);
