from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


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

rag_chain = prompt | llm | StrOutputParser()


