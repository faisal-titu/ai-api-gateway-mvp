def create_face_index(client, index_name: str):
    """
    Creates an OpenSearch index for face embeddings.
    """
    response = client.indices.create(
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
                    },
                    "face_id": {"type": "keyword"},
                    "image_id": {"type": "keyword"}
                }
            }
        }
    )
    return response

