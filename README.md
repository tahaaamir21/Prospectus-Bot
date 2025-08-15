# Prospectus-Bot

An AI-powered chatbot that provides instant answers to questions about NED University using the official prospectus document. Built with LangChain, Groq, and Streamlit.

## 🌟 Features

- **Smart Q&A**: Ask any question about NED University programs, policies, and information
- **Accurate Responses**: Uses RAG (Retrieval-Augmented Generation) for context-aware answers
- **Beautiful UI**: Modern, dark-themed chat interface built with Streamlit
- **Fast Performance**: Powered by Groq's lightning-fast LLM inference
- **No Hallucinations**: Only provides answers based on the actual prospectus content

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Groq API key ([Get one here](https://console.groq.com/))

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Prospectus_bot
   ```

2. **Install dependencies**
   ```bash
   pip install streamlit langchain-community langchain-groq faiss-cpu python-dotenv sentence-transformers
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. **Process the prospectus data** (one-time setup)
   ```bash
   python create_memory.py
   ```

5. **Run the chatbot**
   ```bash
   streamlit run prospectus.py
   ```

6. **Open your browser**
   Navigate to `http://localhost:8501`

## 📁 Project Structure

```
Prospectus_bot/
├── prospectus.py          # Main Streamlit web application
├── create_memory.py       # Data processing pipeline
├── data/                  # Source documents
│   └── UGProspectus2025.pdf
├── vectorstore/           # Processed vector database
│   └── db_faiss/
└── README.md
```

## 🔧 How It Works

### 1. Data Processing (`create_memory.py`)
- Loads PDF documents from the `data/` folder
- Extracts and chunks text into smaller segments
- Creates embeddings using HuggingFace's sentence transformers
- Stores everything in a FAISS vector database

### 2. Question Answering (`prospectus.py`)
- User asks a question through the web interface
- System searches the vector database for relevant context
- Sends context + question to Groq LLM
- Returns accurate, formatted answers

## 🛠️ Technology Stack

- **LangChain**: LLM application framework
- **Groq**: Fast LLM inference API
- **FAISS**: Vector similarity search
- **HuggingFace**: Text embeddings
- **Streamlit**: Web interface
- **PyPDF**: PDF text extraction

## 📝 Usage Examples

Ask questions like:
- "What are the admission requirements for Computer Science?"
- "How much is the tuition fee for undergraduate programs?"
- "What are the available scholarships?"
- "What is the campus address?"
- "Tell me about the library facilities"

## 🔑 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API_KEY` | Your Groq API key | Yes |

## 🚀 Deployment

### Local Development
```bash
streamlit run prospectus.py
```

### Production Deployment
The app can be deployed on:
- Streamlit Cloud
- Heroku
- AWS/GCP/Azure
- Docker containers



## 🔮 Future Enhancements

- [ ] Add support for multiple languages
- [ ] Implement conversation memory
- [ ] Add file upload functionality
- [ ] Create mobile app version
- [ ] Add analytics dashboard
- [ ] Support for more document types

---
