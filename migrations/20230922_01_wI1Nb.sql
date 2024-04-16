-- 
-- depends:
-- Captions
CREATE TABLE IF NOT EXISTS captions_tree (
	id UUID PRIMARY KEY,
	created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS captions_node (
	id UUID PRIMARY KEY,
	tree_id UUID NOT NULL,
	depth INT NOT NULL,
	vec JSONB NOT NULL,
	children JSONB NOT NULL,
	features JSONB NOT NULL,
	CONSTRAINT captions_node_tree_id_fk FOREIGN KEY (tree_id) REFERENCES captions_tree(id)
);
CREATE INDEX IF NOT EXISTS captions_node_depth_tree_id_idx ON captions_node (depth, tree_id);
CREATE TABLE IF NOT EXISTS captions_train_job (
	id UUID PRIMARY KEY,
	node_id UUID NOT NULL,
	CONSTRAINT captions_train_job_node_id_fk FOREIGN KEY (node_id) REFERENCES captions_node(id)
);
CREATE TABLE IF NOT EXISTS captions_tf (
	term_id UUID NOT NULL,
	doc_id UUID NOT NULL,
	frequency INT NOT NULL,
	tree_id UUID NOT NULL,
	CONSTRAINT captions_tf_tree_id_fk FOREIGN KEY (tree_id) REFERENCES captions_tree(id),
	PRIMARY KEY (term_id, doc_id)
);
CREATE TABLE IF NOT EXISTS captions_df (
	term_id UUID PRIMARY KEY,
	unique_docs_count INT NOT NULL,
	total_tf INT NOT NULL,
	tree_id UUID NOT NULL,
	CONSTRAINT captions_df_tree_id_fk FOREIGN KEY (tree_id) REFERENCES captions_tree(id)
);
CREATE TABLE IF NOT EXISTS captions_tree_doc_count (
	tree_id UUID PRIMARY KEY,
	docs_count INT NOT NULL,
	CONSTRAINT captions_tree_doc_count_tree_id_fk FOREIGN KEY (tree_id) REFERENCES captions_tree(id)
);
-- Images
CREATE TABLE IF NOT EXISTS ims_tree (
	id UUID PRIMARY KEY,
	created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS ims_node (
	id UUID PRIMARY KEY,
	tree_id UUID NOT NULL,
	depth INT NOT NULL,
	vec JSONB NOT NULL,
	children JSONB NOT NULL,
	features JSONB NOT NULL,
	CONSTRAINT ims_node_tree_id_fk FOREIGN KEY (tree_id) REFERENCES ims_tree(id)
);
CREATE INDEX IF NOT EXISTS ims_node_depth_tree_id_idx ON ims_node (depth, tree_id);
CREATE TABLE IF NOT EXISTS ims_train_job (
	id UUID PRIMARY KEY,
	node_id UUID NOT NULL,
	CONSTRAINT ims_train_job_node_id_fk FOREIGN KEY (node_id) REFERENCES ims_node(id)
);
CREATE TABLE IF NOT EXISTS ims_tf (
	term_id UUID NOT NULL,
	doc_id UUID NOT NULL,
	frequency INT NOT NULL,
	tree_id UUID NOT NULL,
	CONSTRAINT ims_tf_tree_id_fk FOREIGN KEY (tree_id) REFERENCES ims_tree(id),
	PRIMARY KEY (term_id, doc_id)
);
CREATE TABLE IF NOT EXISTS ims_df (
	term_id UUID PRIMARY KEY,
	unique_docs_count INT NOT NULL,
	total_tf INT NOT NULL,
	tree_id UUID NOT NULL,
	CONSTRAINT ims_df_tree_id_fk FOREIGN KEY (tree_id) REFERENCES ims_tree(id)
);
CREATE TABLE IF NOT EXISTS ims_tree_doc_count (
	tree_id UUID PRIMARY KEY,
	docs_count INT NOT NULL,
	CONSTRAINT ims_tree_doc_count_tree_id_fk FOREIGN KEY (tree_id) REFERENCES ims_tree(id)
);
-- SemSearch service
CREATE TABLE IF NOT EXISTS image_metadata (
	id UUID PRIMARY KEY,
	captions JSONB NOT NULL,
	dataset TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS image_metadata_dataset_idx ON image_metadata (dataset);
