USE embeddings;

CREATE TABLE IF NOT EXISTS vectors (
    `id` String COMMENT 'Unique id of the chunked document embedding',
    `document` String COMMENT 'Contains body of the chunked document',
    `metadata` String COMMENT 'Name of the document',
    `embeddings` Array(Float32) COMMENT 'Document embeddings for similarity search'
) ENGINE = MergeTree()
ORDER BY id;
