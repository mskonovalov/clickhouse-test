import clickhouse_connect


def create_client():
    return clickhouse_connect.get_client(
        host="localhost",
        port=8123,
        username="vectordbuser",
        password="pwd_vectordb",
        database="embeddings",
    )


def main():
    client = create_client()
    result = client.query("SELECT * FROM vectors LIMIT 1")
    print(result.result_rows)


if __name__ == "__main__":
    main()
