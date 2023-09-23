-- 
-- depends: 

CREATE TABLE node (
	id UUID PRIMARY KEY,
	is_root BOOLEAN NOT NULL,
	children JSONB NOT NULL,
	vec BYTEA NOT NULL,
	features JSONB NOT NULL
);

CREATE TABLE image_metadata (
	id UUID PRIMARY KEY,
	label TEXT NOT NULL
);
