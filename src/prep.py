from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import duckdb
import pandas as pd
import os
import dotenv

# Load the environment variables from the .env file
dotenv.load_dotenv()

# List of URLs to load documents from
urls = [
    "https://medium.com/@mehran1414/churn-prediction-in-sparkify-dataset-5e61230c338d",
    "https://medium.com/@mehran1414/automating-stock-market-data-pipeline-with-apache-airflow-minio-spark-and-postgres-b67f7379566a",
    "https://medium.com/@mehran1414/top-poor-hosts-in-seatle-airbnb-dataset-what-factors-are-important-and-why-fd5535f1d96d",
]
# Load documents from the URLs
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

#print(f"Number of documents: {len(docs_list)}")


# Split the documents into sentences
# Initialize a text splitter with specified chunk size and overlap
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=200, chunk_overlap=0
)
# Split the documents into chunks
doc_splits = text_splitter.split_documents(docs_list)


# Generate embeddings for documents using OpenAIEmbeddings
embedding_model = OpenAIEmbeddings(openai_api_key= os.getenv("OPENAI_API_KEY"))
embeddings = embedding_model.embed_documents(doc_splits)

# Step 4: Convert the embeddings and document info to a Pandas DataFrame
document_data = []
for i, doc in enumerate(docs_list):
    document_data.append({
        "id": doc["id"],
        "content": doc["content"],
        "metadata": doc["metadata"],
        "embedding": embeddings[i]
    })

df = pd.DataFrame(document_data)

# Step 5: Store the embeddings in DuckDB
con = duckdb.connect('vector_store.db')

#Create a table for documents with embeddings
con.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY,
        content TEXT,
        metadata TEXT,
        embedding BLOB
    )
""")

# Insert the documents and embeddings into the table
for index, row in df.iterrows():
    # DuckDB requires binary format for the embedding, so we convert it
    embedding_blob = duckdb.blob(row['embedding'])
    con.execute("INSERT INTO documents (id, content, metadata, embedding) VALUES (?, ?, ?, ?)",
                [row['id'], row['content'], row['metadata'], embedding_blob])

# Commit the changes
con.commit()
con.close()

print("Embeddings and documents have been stored in  a vector store with the help of DuckDB successfully!")
