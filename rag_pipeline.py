from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

def build_reference_chain(llm):
    template = """You are a helpful assistant.
Use the following reference to answer the question.

Reference:
{context}

Question:
{question}

Answer in detail:"""
    prompt = PromptTemplate.from_template(template)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain

def build_rag_chain(llm, tasks_C):
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    all_docs = []
    for t in tasks_C:
        for p in t["corpus_passages"]:
            all_docs.append(Document(page_content=p["text"], metadata={"source": p["doc_id"]}))
    db = FAISS.from_documents(all_docs, embedder)
    retriever = db.as_retriever(search_kwargs={"k": 5})
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return rag_chain
