from clickhouse_connect import get_client

client = get_client(
    host="clickhouse",
    port=8123,
    username="default",
    password="",
    database="ivf_mlops"
)
