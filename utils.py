import os
import hashlib
from typing import List
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from PyPDF2 import PdfReader


# ------------------------
# Embeddings
# ------------------------
def get_embeddings():
    """Return HuggingFace sentence transformer embeddings (CPU)."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}  # 👈 Force CPU
    )


# ------------------------
# Load PDF
# ------------------------
def load_pdf(file) -> List[Document]:
    """Extract text from PDF and return list of LangChain Documents."""
    reader = PdfReader(file)
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            docs.append(Document(page_content=text, metadata={"page": i + 1}))
    return docs


# ------------------------
# Split documents safely
# ------------------------
def split_documents(docs, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    return splitter.split_documents(docs)


# ------------------------
# Index documents in FAISS
# ------------------------
def index_pdf_bytes(file, filename: str = "file.pdf", chunk_size: int = 500, chunk_overlap: int = 50):
    docs = load_pdf(file)
    split_docs = split_documents(docs, chunk_size, chunk_overlap)

    embeddings = get_embeddings()
    file_id = hashlib.md5(filename.encode()).hexdigest()
    index_dir = f"faiss_index_{file_id}"

    try:
        if os.path.exists(index_dir):
            db = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=False)
        else:
            db = FAISS.from_documents(split_docs, embeddings)
            db.save_local(index_dir)
    except Exception:
        # fallback in case index is corrupted
        db = FAISS.from_documents(split_docs, embeddings)
        db.save_local(index_dir)

    return db, file_id


# ------------------------
# HuggingFace LLM
# ------------------------
def get_hf_llm(model_name="google/flan-t5-base", max_length=512, device=-1):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        device=device  # -1 = CPU, 0 = GPU
    )
    return HuggingFacePipeline(pipeline=pipe)


# ------------------------
# QA Chain
# ------------------------
def make_qa_chain(llm: HuggingFacePipeline):
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a helpful assistant. "
            "Answer the following question using only the context provided.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\nAnswer:"
        )
    )
    return LLMChain(llm=llm, prompt=prompt)


# ------------------------
# Generate questions + answers
# ------------------------
def generate_questions_from_text(text: str, n_qs: int = 5, llm: HuggingFacePipeline = None):
    prompt = PromptTemplate(
        input_variables=["context", "n_qs"],
        template=(
            "You are given the following text (a resume or document). "
            "Generate exactly {n_qs} unique and meaningful question-answer pairs. "
            "Cover different aspects such as summary, education, internships, projects, "
            "skills, achievements, and future goals. Avoid repeating the same question.\n\n"
            "Text:\n{context}\n\n"
            "Return your output in this numbered format:\n"
            "1. Question: ...\n   Answer: ...\n"
            "2. Question: ...\n   Answer: ...\n"
            "...\n"
            "{n_qs}. Question: ...\n   Answer: ..."
        )
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.invoke({"context": text, "n_qs": n_qs})
    return result["text"] if isinstance(result, dict) else result
