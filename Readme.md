# Empire RAG Test

A simple RAG (Retrieval-Augmented Generation) application using Empire Chain for document processing and question answering.

## Features

- PDF document processing and vectorization
- Text and audio query support
- Qdrant vector store for similarity search
- OpenAI embeddings for text representation
- Groq LLM for text generation
- Groq STT for speech-to-text conversion

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/manas95826/empire-rag-test.git
cd empire-rag-test
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the project root with the following keys:

```env
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Prepare your document

Place a PDF file named `input.pdf` in the project directory.

### 5. Run the application

```bash
python app.py
```

## Usage

The application will:
1. Load and process the `input.pdf` file
2. Create embeddings and store them in a vector database
3. Ask a predefined question about the document
4. Retrieve relevant context and generate an answer

## Audio Input (Optional)

To use audio input instead of text queries, modify the main function call:

```python
if __name__ == "__main__":
    main(if_audio_input=True)
```

Make sure to place an `audio.mp3` file in the project directory when using audio input.

## Dependencies

- `empire-chain`: Core RAG framework
- `sentence-transformers`: Text embeddings
- `python-dotenv`: Environment variable management

## API Keys Required

- **OpenAI API Key**: For text embeddings
- **Groq API Key**: For LLM and STT services
- **Qdrant**: For vector storage (can use local instance)
