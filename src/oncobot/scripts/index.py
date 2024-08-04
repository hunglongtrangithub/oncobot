import os
import json
import requests
from uuid import uuid4
from bson import ObjectId

import meilisearch

from src.utils.env_config import settings
from src.utils.logger_config import get_logger

logger = get_logger(__name__)

EMBEDDER_NAME = "default"
INDEX_NAME = "clinical_docs"
EMBEDDINGS_MODEL_NAME = "BAAI/bge-base-en-v1.5"
MEILI_API_URL = "http://" + (settings.meili_http_addr or "localhost:7700")


def get_api_keys():
    master_client = meilisearch.Client(
        url=MEILI_API_URL,
        api_key=settings.meili_master_key.get_secret_value(),
    )

    search_actions = ["search"]
    search_indexes = [INDEX_NAME]
    admin_actions = ["*"]
    admin_indexes = ["*"]
    search_key = None
    admin_key = None

    # Fetch all existing keys
    keys = master_client.get_keys().results

    # Check if keys with the specified actions and indexes exist
    for key in keys:
        if search_key and admin_key:
            break
        elif key.actions == search_actions and key.indexes == search_indexes:
            logger.info("Found existing search key.")
            search_key = key
        elif key.actions == admin_actions and key.indexes == admin_indexes:
            logger.info("Found existing admin key.")
            admin_key = key

    # Create keys if they do not exist
    if not search_key:
        logger.info("Creating search key...")
        search_uid = str(uuid4())
        search_key = master_client.create_key(
            options={
                "uid": search_uid,
                "description": "API key for the clinical docs search app",
                "actions": search_actions,
                "indexes": search_indexes,
                "expiresAt": None,
            }
        )

    if not admin_key:
        logger.info("Creating admin key...")
        admin_uid = str(uuid4())
        admin_key = master_client.create_key(
            options={
                "uid": admin_uid,
                "description": "API key for the clinical docs admin app",
                "actions": admin_actions,
                "indexes": admin_indexes,
                "expiresAt": None,
            }
        )

    return search_key.key, admin_key.key


SEARCH_API_KEY, ADMIN_API_KEY = get_api_keys()


def process_docs():
    docs_folder = "docs"
    json_file_path = f"{docs_folder}/processed_docs.jsonl"

    with open(json_file_path, mode="w", encoding="utf-8") as json_file:
        docs = []
        for patient_folder in os.listdir(docs_folder):
            patient_path = os.path.join(docs_folder, patient_folder)
            if os.path.isdir(patient_path):
                for txt_file in os.listdir(patient_path):
                    if txt_file.endswith(".txt"):
                        file_path = os.path.join(patient_path, txt_file)
                        with open(file_path, "r", encoding="utf-8") as f:
                            page_content = f.read()

                        document = {
                            "id": str(ObjectId()),
                            "title": txt_file,
                            "source": f"{patient_folder}/{txt_file}",
                            "page_content": page_content,
                        }
                        json_file.write(json.dumps(document) + "\n")
                        docs.append(document)
                        print(document["source"])
        logger.info(f"Processed {len(docs)} documents.")
        logger.info(f"Documents processed and saved to {json_file_path}.")
        return docs


def index_docs_to_meili(docs):
    admin_client = meilisearch.Client(url=MEILI_API_URL, api_key=ADMIN_API_KEY)

    current_indexes = [index.uid for index in admin_client.get_indexes()["results"]]
    if INDEX_NAME not in current_indexes:
        logger.info(f"Index {INDEX_NAME} does not yet exist. Creating...")
        task = admin_client.create_index(uid=INDEX_NAME, options={"primaryKey": "id"})
        admin_client.wait_for_task(task.task_uid)
        logger.info(f"Index {INDEX_NAME} created.")

    index = admin_client.index(INDEX_NAME)

    # enable vector search
    data = {"vectorStore": True}
    response = requests.patch(
        url=f"{MEILI_API_URL}/experimental-features",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {ADMIN_API_KEY}",
        },
        data=json.dumps(data),
    )
    if response.status_code == 200:
        print("Vector search enabled successfully!")
    else:
        print("Error enabling vector search:", response.status_code)
        return

    # Clear the index before adding new documents
    delete_task = index.delete_all_documents()
    index_task = index.add_documents(docs)

    # add the embedder
    settings = {
        "embedders": {
            EMBEDDER_NAME: {
                "source": "huggingFace",
                "model": EMBEDDINGS_MODEL_NAME,
                "documentTemplate": "A clinical document titled {{doc.title}} whose content is {{doc.page_content}}",
            }
        },
        "filterableAttributes": ["title", "source"],
        "sortableAttributes": ["title", "source"],
        "searchableAttributes": ["title", "source", "page_content"],
    }

    tasks = {
        "delete_all_documents": delete_task,
        "add_documents": index_task,
        "update_embedders": index.update_embedders(settings["embedders"]),
        "update_filterable_attributes": index.update_filterable_attributes(
            settings["filterableAttributes"]
        ),
        "update_sortable_attributes": index.update_sortable_attributes(
            settings["sortableAttributes"]
        ),
        "update_searchable_attributes": index.update_searchable_attributes(
            settings["searchableAttributes"]
        ),
    }

    return tasks


def index_docs():
    docs = process_docs()
    tasks = index_docs_to_meili(docs)
    if not tasks:
        return

    for task_name, task in tasks.items():
        print(f"{task_name} task status: {task.status}")


def test_search_meili():
    index = meilisearch.Client(
        url=MEILI_API_URL,
        api_key=SEARCH_API_KEY,
    ).index(INDEX_NAME)

    def search_documents(query):
        opt_params = {
            "hybrid": {
                "semanticRatio": 0.5,
                "embedder": EMBEDDER_NAME,
            },
            "limit": 5,
        }
        return index.search(query, opt_params)["hits"]

    # Example query
    query = "Fake Patient1"
    documents = search_documents(query)
    print(f"Found {len(documents)} documents for query: {query}")
    # Print the results
    for doc in documents:
        print(f"ID: {doc['id']}")
        print(f"Title: {doc['title']}")
        print(f"Source: {doc['source']}")
        print("-" * 40)


if __name__ == "__main__":
    index_docs()
    # test_search_meili()
    # get_api_keys()
    # process_docs()
    pass
