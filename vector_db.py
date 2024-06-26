import chromadb

client = chromadb.Client()

collection = client.create_collection(name="my_collection")

collection.add(
    documents=[
        "Hello, how are you?",
        "I am fine, thank you.",
    ],
    ids=['id1', 'id2'],
)

all_docs = collection.get()

print(all_docs)

documents = collection.get(ids=["id1"])

print(documents)