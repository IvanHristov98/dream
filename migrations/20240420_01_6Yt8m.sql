-- 
-- depends: 20230922_01_wI1Nb
-- Captions store
CREATE TABLE IF NOT EXISTS image_metadata_sample_log (
    im_id UUID NOT NULL,
    tree_id UUID NOT NULL,
    doc_store TEXT NOT NULL,
    sampled_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT sampled_image_metadata_im_id_fk FOREIGN KEY (im_id) REFERENCES image_metadata(id),
    PRIMARY KEY (im_id, tree_id)
);