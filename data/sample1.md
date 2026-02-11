# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) is an AI framework that enhances large language model (LLM) outputs by incorporating external knowledge retrieval into the generation process. Instead of relying solely on the knowledge encoded during training, RAG systems first retrieve relevant documents from a knowledge base and then use those documents as context for generating responses.

## How RAG Works

The RAG pipeline consists of three main stages:

1. **Indexing**: Documents are preprocessed, split into chunks, and converted into vector embeddings. These embeddings are stored in a vector database for efficient similarity search.

2. **Retrieval**: When a user asks a question, the query is also converted into a vector embedding. The system then searches the vector database for the most similar document chunks using cosine similarity or other distance metrics.

3. **Generation**: The retrieved document chunks are combined with the original question and passed to an LLM as context. The LLM then generates an answer that is grounded in the retrieved information.

## Benefits of RAG

- **Reduced hallucinations**: By grounding responses in actual documents, RAG significantly reduces the tendency of LLMs to generate factually incorrect information.
- **Up-to-date knowledge**: The knowledge base can be updated without retraining the model, allowing the system to provide current information.
- **Source attribution**: RAG systems can cite specific sources for their answers, improving transparency and trust.
- **Domain specialization**: Organizations can build RAG systems over their proprietary documents to create domain-specific AI assistants.

## Vector Databases

Vector databases are purpose-built storage systems optimized for similarity search over high-dimensional vectors. Popular options include:

- **Qdrant**: A high-performance vector database written in Rust, offering both in-memory and on-disk storage with rich filtering capabilities.
- **Pinecone**: A fully managed cloud-native vector database service.
- **Weaviate**: An open-source vector database with built-in vectorization modules.
- **ChromaDB**: A lightweight, developer-friendly embedding database.

## Embedding Models

Embedding models convert text into dense vector representations that capture semantic meaning. Common choices include:

- **Google Gemini Embedding API**: Provides high-quality embeddings through a simple API call, with models like `gemini-embedding-exp-03-07`.
- **OpenAI Embeddings**: The `text-embedding-3-small` and `text-embedding-3-large` models offer different quality/cost tradeoffs.
- **Sentence Transformers**: Open-source models like `all-MiniLM-L6-v2` that can run locally without API calls.

## Chunking Strategies

How documents are split into chunks significantly impacts retrieval quality:

- **Fixed-size chunking**: Simple approach that splits text into chunks of a fixed number of tokens with overlap.
- **Semantic chunking**: Uses sentence boundaries or paragraph breaks to create more meaningful chunks.
- **Recursive chunking**: Tries multiple splitting strategies (paragraphs, sentences, words) to achieve target chunk sizes.
- **Document-aware chunking**: Respects document structure like headings, lists, and code blocks.
