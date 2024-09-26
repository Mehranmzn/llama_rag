from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
    chunk_size=250, chunk_overlap=0
)
# Split the documents into chunks
doc_splits = text_splitter.split_documents(docs_list)