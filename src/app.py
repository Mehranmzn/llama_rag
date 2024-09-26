from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ragclass import RagPrompt
from retriever import DuckDBRetriever
import streamlit as st



# Define the prompt template for the LLM
prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks.
    Use the following documents or urls to answer the question.
    If you don't know the answer, just say that you don't know, nothing more is required.
    Use three sentences maximum and keep the answer concise:
    Question: {question}
    Documents: {docs}
    Answer:
    """,
    input_variables=["question", "docs"],
)


llm = ChatOllama(
    model="llama3.1",
    temperature=0,
)


parser = StrOutputParser()
rag_chain = prompt | llm | parser
retriever = DuckDBRetriever(db_path='documents_with_embeddings.db', k=4)
rag_application = RagPrompt(retriever, rag_chain)


question = st.text_input("Enter your question:", "")

if st.button("Get Answer"):
    if question:
        # Step 8: Run the RAG application and get the answer
        with st.spinner("Retrieving documents and generating answer..."):
            answer = rag_application.run(question)

        # Step 9: Display the retrieved documents
        st.subheader("Retrieved Documents:")
        top_k_documents = retriever.retrieve(embedding_model.embed_query(question))

        for idx, doc in enumerate(top_k_documents):
            st.write(f"**Document {idx + 1}:** {doc['content']}")
            st.caption(f"Metadata: {doc['metadata']}")

        # Step 10: Display the final answer
        st.subheader("Answer:")
        st.write(answer)
    else:
        st.error("Please enter a question!")
        




