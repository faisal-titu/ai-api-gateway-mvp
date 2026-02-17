import os
from opensearchpy import OpenSearch

host = [{'host': os.getenv('OPENSEARCH_HOST', 'opensearch-node1'), 'port': int(os.getenv('OPENSEARCH_PORT', 9200))}]

def get_client():

    client = OpenSearch(
        hosts=host,
        http_compress=True,
        # http_auth=auth,
        use_ssl=False,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
        timeout=120,
        max_retries=3,
        retry_on_timeout=True,
    )
    return client

client = get_client()
