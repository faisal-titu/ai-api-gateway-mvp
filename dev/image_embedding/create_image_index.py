# from open_search_client import client


def create_index(client, index_name):
    # Create the index
    response=client.indices.create(
        index=index_name,
        body={
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100
                }
            },
            "mappings": {
                "properties": {
                    "my_vector": {
                        "type": "knn_vector",
                        "dimension": 512,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "lucene",
                            "parameters": {
                                "ef_construction": 128,
                                "m": 24
                            }
                        }
                    }
                }
            }
        }
    )
    return response