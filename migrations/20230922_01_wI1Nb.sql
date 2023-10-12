-- 
-- depends: 

CREATE TABLE IF NOT EXISTS node (
	id UUID PRIMARY KEY,
	is_root BOOLEAN NOT NULL,
	children JSONB NOT NULL,
	vec BYTEA NOT NULL,
	features JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS image_metadata (
	id UUID PRIMARY KEY,
	labels JSONB NOT NULL,
	dataset TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS image_metadata_dataset ON image_metadata (dataset);
