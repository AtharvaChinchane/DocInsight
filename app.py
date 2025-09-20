import streamlit as st
from utils import index_pdf_bytes, make_qa_chain, generate_questions_from_text, get_hf_llm

st.set_page_config(page_title="📄 PDF RAG App", layout="wide")
st.title("📄 PDF Question Answering & Question Generation")

# ------------------------
# Upload PDF
# ------------------------
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    st.info("Indexing PDF...")
    vectorstore, file_id = index_pdf_bytes(uploaded_file, filename=uploaded_file.name)
    st.success(f"Indexed: {uploaded_file.name}")

    # ------------------------
    # Load LLM (cached)
    # ------------------------
    @st.cache_resource
    def load_llm():
        return get_hf_llm(model_name="google/flan-t5-base", max_length=512, device=-1)

    llm = load_llm()
    st.success("LLM ready ✅")

    # ------------------------
    # Ask a Question
    # ------------------------
    st.subheader("Ask a Question about the PDF")
    question = st.text_input("Enter your question:")
    if question:
        qa_chain = make_qa_chain(llm)
        docs = vectorstore.similarity_search(question, k=3)
        context = "\n".join([d.page_content for d in docs])

        with st.spinner("Generating answer..."):
            result = qa_chain.invoke({"context": context, "question": question})

        st.subheader("Answer")
        st.write(result["text"] if isinstance(result, dict) else result)

    # ------------------------
    # Generate Questions
    # ------------------------
    st.subheader("Generate Questions + Answers from PDF content")
    n_qs = st.number_input("Number of questions:", min_value=1, max_value=20, value=5)

    if st.button("Generate Questions"):
        raw_text = "\n".join([d.page_content for d in vectorstore.docstore._dict.values()])
        with st.spinner("Generating questions and answers..."):
            out = generate_questions_from_text(raw_text, n_qs=n_qs, llm=llm)
        st.markdown(out)
