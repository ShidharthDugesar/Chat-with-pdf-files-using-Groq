import streamlit as st
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from groq import Groq

# Streamlit app title
st.title("Document Query Finder")
st.write("Upload a PDF or DOCX file and ask questions based on its content.")

# File uploader (PDF and DOCX)
uploaded_file = st.file_uploader("Upload a PDF or DOCX file", type=["pdf", "docx"])

# Initialize memory in session state
#Initialize values in Session State
## its Stores contextual memory for the AI
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# Initialize conversation history in session state
## Stores the full chat history for display/logging in list or dict like this {"role": "user", "content": "Hello"})
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if uploaded_file is not None:
    # Save the uploaded file temporarily
    file_path = f"./{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load the document based on file type
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        st.error("Unsupported file format. Please upload a PDF or DOCX file.")
        st.stop()

    # Load and split the document
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    all_splits = text_splitter.split_documents(data)

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Create a vector store
    vector_store = FAISS.from_documents(all_splits, embeddings)
    vector_store.save_local("faiss_index")

    # Load the vector store
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Initialize Groq client
    groq_client = Groq(api_key="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")

    # Define the prompt template
    template = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Chat History:
    {history}

    Human: {input}
    AI:"""

    prompt = PromptTemplate.from_template(template)

    # Create the RetrievalQA chain with hybrid retrieval
    def hybrid_qa_chain(query, retriever, memory):
        # Check if the query is about personal details (e.g., name, location)
        personal_keywords = ["name", "location", "from"]
        if any(keyword in query.lower() for keyword in personal_keywords):
            # Use only the conversation history for personal questions
            context = "No additional context from the document is needed for this question."
        else:
            # Retrieve from PDF for other questions
            pdf_results = retriever.get_relevant_documents(query)
            context = "\n".join([doc.page_content for doc in pdf_results]) if pdf_results else "No relevant context found."

        # Generate response using Groq
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": prompt.format(context=context, question=query, history=memory.buffer, input=query)},
                {"role": "user", "content": query}
            ]
        )
        return response.choices[0].message.content

    # Query input
    query = st.text_input("Ask a question about the document:")
    if query:
        # Generate a response using the hybrid QA chain
        response = hybrid_qa_chain(query, vector_store.as_retriever(), st.session_state.memory)
        
        # Save the conversation to memory
        st.session_state.memory.save_context({"input": query}, {"output": response})
        
        # Update conversation history
        st.session_state.conversation_history.append({"role": "Human", "content": query})
        st.session_state.conversation_history.append({"role": "AI", "content": response})

# Display conversation history
if st.session_state.conversation_history:
    st.write("### Conversation History:")
    for message in st.session_state.conversation_history:
        if message["role"] == "Human":
            st.write(f"**You:** {message['content']}")
        else:
            st.write(f"**AI:** {message['content']}")