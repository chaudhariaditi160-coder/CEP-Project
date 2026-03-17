import streamlit as st
from src.pdf_loader import load_pdf
from src.retriever import chunk_text, create_index, retrieve
from src.qa_model import get_answer

# Page config
st.set_page_config(page_title="Knowledge QA System", page_icon="📄")

st.title("📄 Knowledge Driven QA System")
st.write("Upload a PDF and ask questions from it.")

# Upload PDF
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:

    st.success("✅ PDF uploaded successfully!")

    # Process PDF
    with st.spinner("Processing PDF..."):
        text = load_pdf(uploaded_file)
        chunks = chunk_text(text)
        index, embeddings = create_index(chunks)

    st.success("✅ PDF processed successfully!")

    # Ask question
    question = st.text_input("💬 Ask a question from the document")

    if question:

        with st.spinner("Finding answer..."):
            retrieved_chunks = retrieve(question, chunks, index)
            context = " ".join(retrieved_chunks)
            answer = get_answer(question, context)

        st.write("### 📌 Answer:")
        st.write(answer)

else:
    st.info("📂 Please upload a PDF file to begin.")
