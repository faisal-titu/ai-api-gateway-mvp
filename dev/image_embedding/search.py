import random
from . import open_search_client

def search_knn(query_embedding, index_name, num_images):
    # Search the embeddings
    response = open_search_client.client.search(
        index=index_name,
        body={
            "size": num_images,
            "query": {
                "knn": {
                    "my_vector": {
                        "vector": query_embedding,
                        "k": num_images
                    }
                }
            }
        }
    )
    # Log the OpenSearch response
    # print(f"OpenSearch response: {response}")
    print(f"Total hits returned: {response['hits']['total']['value']}")  # Total number of hits
    print(f"Number of hits in 'hits': {len(response['hits']['hits'])}")  # Number of hits processed
    # print(f"Number of hits returned: {len(response['hits']['hits'])}")
    return response


def get_random_imageVectors_with_IDs(index_name):

    # Get the total number of documents in the index
    total_docs = open_search_client.client.count(index=index_name)["count"]
    
    # Generate a random offset within the range of total documents
    random_offset = random.randint(0, total_docs - 1)

    # Search for a single document using the random offset
    response = open_search_client.client.search(
        index=index_name,
        body={
            "query": {"match_all": {}},  # Match all documents
            "size": 1,  # Retrieve only one document
            "from": random_offset  # Use the random offset
        }
    )

    # Extract the vector and image ID from the retrieved document
    if response["hits"]["hits"]:
        # Get the vector field name (e.g., "my_vector")
        vector_field_name = next(iter(response["hits"]["hits"][0]["_source"]))
        # Extract the vector value
        vector = response["hits"]["hits"][0]["_source"][vector_field_name]
        # Extract the image ID
        image_id = response["hits"]["hits"][0]["_source"]["image_id"]

    return image_id , vector