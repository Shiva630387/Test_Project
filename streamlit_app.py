import streamlit as st
import tempfile
import hashlib

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI


# --------------------------------
# PAGE CONFIG
# --------------------------------

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Document Assistant")
st.caption("Upload a PDF and chat with your document")


# --------------------------------
# SESSION STATE
# --------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "file_hash" not in st.session_state:
    st.session_state.file_hash = None


# --------------------------------
# SIDEBAR
# --------------------------------

with st.sidebar:

    st.header("📂 Document")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# --------------------------------
# PROCESS NEW PDF
# --------------------------------

if uploaded_file is not None:

    # Create unique ID for uploaded PDF
    current_file_hash = hashlib.md5(
        uploaded_file.getvalue()
    ).hexdigest()

    # Process only if a NEW PDF is uploaded
    if current_file_hash != st.session_state.file_hash:

        with st.spinner("Processing new PDF..."):

            # Reset old data
            st.session_state.messages = []
            st.session_state.vector_db = None

            # Save PDF temporarily
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getbuffer()
                )

                temp_path = temp_file.name


            # Load PDF
            loader = PyPDFLoader(temp_path)

            documents = loader.load()


            # Split document
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = text_splitter.split_documents(
                documents
            )


            # Create embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )


            # Create vector database
            vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings
            )


            # Save in session
            st.session_state.vector_db = vector_db

            st.session_state.file_hash = current_file_hash


        st.sidebar.success("PDF processed successfully! 🎉")

        st.sidebar.info(
            f"Pages: {len(documents)}\n\n"
            f"Chunks: {len(chunks)}"
        )


# --------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])

        # Show sources for assistant messages
        if (
            message["role"] == "assistant"
            and "sources" in message
        ):

            with st.expander("📚 View Sources"):

                for source in message["sources"]:

                    st.write(
                        f"📄 Page {source['page']}"
                    )

                    st.write(source["content"])

                    st.divider()


# --------------------------------
# CHAT INPUT
# --------------------------------

if prompt := st.chat_input(
    "Ask something about your PDF..."
):

    if st.session_state.vector_db is None:

        st.warning("⚠️ Please upload a PDF first.")

    else:

        # Save user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })


        # Show user message
        with st.chat_message("user"):

            st.write(prompt)


        # AI RESPONSE
        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                # Retrieve relevant chunks
                results = (
                    st.session_state.vector_db
                    .similarity_search(
                        prompt,
                        k=3
                    )
                )


                # Combine retrieved context
                context = "\n\n".join(
                    document.page_content
                    for document in results
                )


                # Create Gemini model
                llm = ChatGoogleGenerativeAI(
                    model="gemini-3.6-flash"
                )


                # RAG prompt
                rag_prompt = f"""
You are an AI Document Assistant.

Answer the user's question using ONLY the
information provided in the document context.

If the answer cannot be found in the document,
say exactly:

"I could not find this information in the document."

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{prompt}
"""


                # Get AI response
                response = llm.invoke(
                    rag_prompt
                )


                # Clean response
                if isinstance(response.content, list):

                    answer = response.content[0]["text"]

                else:

                    answer = response.content


                # Display answer
                st.write(answer)


                # --------------------------------
                # CREATE SOURCE INFORMATION
                # --------------------------------

                sources = []

                for document in results:

                    page_number = (
                        document.metadata.get(
                            "page",
                            0
                        ) + 1
                    )

                    sources.append({
                        "page": page_number,
                        "content": document.page_content
                    })


                # Show sources
                with st.expander("📚 View Sources"):

                    for source in sources:

                        st.write(
                            f"📄 Page {source['page']}"
                        )

                        st.write(
                            source["content"]
                        )

                        st.divider()


        # Save assistant message + sources
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })