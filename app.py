import streamlit as st

from pdf_processor import extract_text
from rag import create_database, ask_question


st.set_page_config(
    page_title="Mini RAG",
    page_icon="📚"
)


st.title("📚 AI PDF Knowledge Assistant")

st.write(
    "Upload a PDF and ask questions about its content."
)


uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


if uploaded_file:

    file_path = "uploads/" + uploaded_file.name

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF uploaded successfully!")

    if st.button("Process PDF"):

        with st.spinner("Processing PDF..."):

            text = extract_text(file_path)

            if not text.strip():

                st.error(
                    "Could not extract text from this PDF."
                )

            else:

                db = create_database(text)

                st.session_state["db"] = db

                st.success(
                    "PDF processed successfully!"
                )


if "db" in st.session_state:

    st.divider()

    st.subheader("Ask a Question")

    question = st.text_input(
        "Enter your question:"
    )


    if st.button("Ask Question"):

        if question.strip():

            with st.spinner("Searching document..."):

                answer = ask_question(
                    st.session_state["db"],
                    question
                )

            st.subheader("Answer")

            st.write(answer)

        else:

            st.warning(
                "Please enter a question."
            )
