from opensearchpy import OpenSearch


def create_index_text(index_name: str, os_client: OpenSearch, dimension: int = 512):
    """Create index in OpenSearch with specified mapping."""
    mapping = {
        "mappings": {
            "properties": {
                "text_id": {"type": "keyword"},
                "description_vector": {
                    "type": "knn_vector",
                    "dimension": dimension,
                },
                "text_field": {
                    "type": "text",
                    "analyzer": "standard",
                    "fields": {
                        "keyword_field": {"type": "keyword"}
                    }
                }
            }
        },
        "settings": {
            "index": {
                "number_of_shards": 1,
                "knn": "true",
                "number_of_replicas": 0
            }
        }
    }
    # os_client.indices.create(index=index_name, body=mapping)
    if not os_client.indices.exists(index=index_name):
        os_client.indices.create(index=index_name, body=mapping)
        print(f"Index {index_name} created.")
    else:
        print(f"Index {index_name} already exists.")

    print(f"Index {index_name} created with dimension {dimension}.")