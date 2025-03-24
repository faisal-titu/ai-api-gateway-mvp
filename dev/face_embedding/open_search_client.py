from opensearchpy import OpenSearch

# Define the OpenSearch host configuration
host = [{'host': 'localhost', 'port': 9200}]

def get_client():
    client = OpenSearch(
        hosts=host,
        http_compress=True,
        use_ssl=False,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False
    )
    return client

client = get_client()
