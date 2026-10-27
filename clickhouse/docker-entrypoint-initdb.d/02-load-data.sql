USE embeddings;

SET max_http_get_redirects = 1;
SET enable_url_encoding = 0;

INSERT INTO vectors (id, document, metadata, embeddings)
SELECT id, document, metadata, embeddings
FROM url(
    'https://huggingface.co/datasets/ClickHouse/software-engineer-python-interview-dataset/resolve/main/data/train-00000-of-00001.parquet',
    'Parquet'
);
