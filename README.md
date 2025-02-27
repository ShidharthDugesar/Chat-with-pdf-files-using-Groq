# Chat-with-pdf-files-using-Groq# Chat with PDF using RAG



This project allows you to chat with a PDF file using **Retrieval-Augmented Generation (RAG)**. It extracts text from the PDF, retrieves relevant information, and generates accurate answers to your questions. The project leverages **Groq** for fast and efficient inference.

---

## Features
- **PDF Text Extraction**: Extract text from uploaded PDF files.
- **Retrieval-Augmented Generation (RAG)**: Retrieve relevant information and generate answers using a generative model.
- **User-Friendly Interface**: Simple and intuitive interface for chatting with PDFs.
- **Fast Inference**: Powered by **Groq** for high-speed AI processing.

---

## How It Works
1. **Upload a PDF**: The system extracts text from the PDF and splits it into smaller chunks.
2. **Generate Embeddings**: Each chunk is converted into embeddings using a pre-trained model.
3. **Retrieve Relevant Information**: When you ask a question, the system retrieves the most relevant chunks using a vector database.
4. **Generate Answers**: The retrieved chunks are passed to a generative model (e.g., GPT or Groq-integrated model) to produce a response.
5. **Display Results**: The system displays the generated answer to the user.

---

## Setup

### Prerequisites
- Python 3.8+
- [Groq API Key](https://groq.com/) 
- [FAISS] ( for vector storage)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/chat-with-pdf-using-RAG.git
   cd chat-with-pdf-using-RAG
