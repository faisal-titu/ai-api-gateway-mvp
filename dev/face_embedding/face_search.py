from dev.image_embedding.open_search_client import client
from typing import Dict, Any, List

def perform_knn_search(index_name: str,embeddings: List[Dict[str, Any]], k: int) -> List[List[Dict[str, Any]]]:
    results = []

    try:
        for embedding in embeddings:
            query_vector = embedding["embedding"]
            knn_query = {
                "size": k,
                "query": {
                    "knn": {
                        "my_vector": {
                            "vector": query_vector,
                            "k": k
                        }
                    }
                }
            }

            # Query OpenSearch
            response = client.search(index=index_name, body=knn_query)
            hits = response["hits"]["hits"]

            # Collect results for this embedding
            face_results = [
                {
                    "image_id": hit["_source"].get("image_id"),
                    "face_id": hit["_source"].get("face_id"),
                    "score": hit["_score"]
                }
                for hit in hits
            ]
            results.append(face_results)

    except Exception as e:
        raise RuntimeError(f"Error during k-NN search: {e}")

    return results
