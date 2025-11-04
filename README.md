# AI Assistant (RAG with Gemini, Langchain & PGVector)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A Dockerized Retrieval-Augmented Generation (RAG) application that uses Google Gemini, Langchain, PGVector, and Hugging Face embeddings to answer questions based on documents provided in a local knowledge base. This specific implementation is configured to act as an AI assistant for the "1337 Coding School".

## Table of Contents

1.  [Overview](#overview)
2.  [Features](#features)
3.  [Technology Stack](#technology-stack)
4.  [Project Structure](#project-structure)
5.  [Setup and Installation](#setup-and-installation)
6.  [Usage](#usage)
7.  [Configuration](#configuration)
8.  [Knowledge Base](#knowledge-base)
9.  [Makefile Commands](#makefile-commands)
10. [License](#license)

## Overview

This project implements a RAG pipeline:

1.  **Load:** Reads PDF documents from a specified directory (`rag/knowledge_base`).
2.  **Split:** Breaks down the documents into smaller, manageable chunks.
3.  **Embed:** Converts text chunks into numerical vectors using a Hugging Face Sentence Transformer model.
4.  **Store:** Stores these embeddings in a PostgreSQL database with the PGVector extension for efficient similarity searching.
5.  **Retrieve:** Given a user query, finds the most relevant document chunks from the vector store.
6.  **Generate:** Uses Google's Gemini model via Langchain, providing the retrieved context and chat history, to generate a relevant and constrained answer.

The assistant is specifically prompted to answer only questions related to "1337 Coding School" based on the provided context, and to respond with "I don't know..." for unrelated queries.

The entire application is packaged within a Docker container for ease of deployment and dependency management.

## Features

*   **Retrieval-Augmented Generation (RAG):** Leverages local documents to provide context-aware answers.
*   **PDF Document Support:** Ingests knowledge from PDF files.
*   **Efficient Vector Storage:** Uses PostgreSQL with PGVector for fast similarity searches.
*   **Persistent Chat History:** Stores conversations in the database to maintain context across sessions.
*   **State-of-the-Art Embeddings:** Utilizes Hugging Face Sentence Transformers (`all-MiniLM-L6-v2`).
*   **Powerful LLM:** Integrates with Google Gemini (`gemini-2.0-flash-exp` by default) via Langchain.
*   **Conversational Context:** Maintains chat history for more natural interactions.
*   **Constrained Responses:** Specifically tailored to answer questions about "1337 Coding School".
*   **Dockerized:** Easy setup and consistent environment using Docker and Docker Compose.
*   **Interactive CLI:** Simple command-line interface for chatting with the assistant.

## Technology Stack

*   **Language:** Python 3.11
*   **Database:** PostgreSQL with PGVector
*   **LLM:** Google Gemini (via `google-generativeai` and `langchain-google-genai`)
*   **Framework:** Langchain
*   **Embeddings:** Hugging Face Sentence Transformers (`sentence-transformers`, `langchain-huggingface`)
*   **Vector Store:** PGVector (`langchain-postgres`)
*   **Document Loading:** `langchain-community` (PyPDFLoader)
*   **Containerization:** Docker, Docker Compose
*   **CLI Interaction:** `pyinputplus`
*   **Configuration:** `python-dotenv`

## Project Structure

```
ai-assistant/
├── README.md             # This file
├── Dockerfile            # Instructions to build the Docker image
├── docker-compose.yml    # Docker Compose configuration
├── env_exemple           # Example environment file structure
├── LICENSE               # Project license (MIT)
├── Makefile              # Commands for building, running, cleaning
├── .dockerignore         # Specifies files/dirs to ignore in Docker build context
└── rag/                  # Core RAG application code
    ├── main.py           # Main Python script with RAG logic
    ├── config.py         # Application configuration
    ├── database.py       # Database connection and initialization
    ├── document_processing.py # Document loading and splitting
    ├── llm.py            # Language model configuration
    ├── vector_store.py   # Vector store logic
    ├── chat.py           # Chat logic
    ├── requirements.txt  # Python dependencies
    └── knowledge_base/   # Directory to place your PDF documents (Mounted volume)
```

## Setup and Installation

**Prerequisites:**

*   Docker ([Install Docker](https://docs.docker.com/engine/install/))
*   Docker Compose ([Install Docker Compose](https://docs.docker.com/compose/install/))
*   Make
*   Git

**Steps:**

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/abizyane/Ai-Assistant
    cd ai-assistant
    ```

2.  **Create Environment File:**
    Copy the example environment file and add your Google Gemini API Key.
    ```bash
    cp env_exemple .env
    nano .env
    ```
    Replace `'your_gemini_api_key'` with your actual key:
    ```env
    GEMINI_API_KEY=AIzaxxxxxxxxxxxxxxxxxxxxxxxxxxx
    ```

3.  **Prepare Knowledge Base:**
    *   Create the knowledge base directory if it doesn't exist:
        ```bash
        mkdir -p rag/knowledge_base
        ```
    *   Place your relevant PDF documents (e.g., information about 1337 Coding School) inside the `rag/knowledge_base` directory. The application will recursively search for `.pdf` files in this directory.

4.  **Build the Docker Image:**
    Use the Makefile to build the image.
    ```bash
    make build
    ```

## Usage

1.  **Run the Application:**
    Use the Makefile to start the interactive container. This command mounts the `knowledge_base` directory into the container.
    ```bash
    make up
    ```

2.  **Interact with the Assistant:**
    *   The application will start, potentially processing documents and creating/loading the vector store (this might take time on the first run or if new documents are added).
    *   You will see a prompt like `You: `. Type your questions about 1337 Coding School.
    *   The assistant will respond based on the documents in the `knowledge_base` and its specific instructions.
    *   Type `exit`, `quit`, or `bye` to end the session.
    *   Press `Ctrl+C` to interrupt and exit.

## Configuration

*   **`GEMINI_API_KEY`**: (Required) Your API key for Google Gemini. Set this in the `.env` file.
*   **`DATABASE_URL`**: (Optional) The connection URL for the PostgreSQL database. Defaults to `postgresql://user:password@db:5432/ragdb`.
*   **`rag/knowledge_base`**: The directory where you must place your source PDF documents.
*   **Constants in `rag/config.py`**: You can modify defaults like `DEFAULT_CHUNK_SIZE`, `DEFAULT_CHUNK_OVERLAP`, `DEFAULT_TEMPERATURE`, `EMBEDDING_MODEL`, `LLM_MODEL` directly in the `rag/config.py` script if needed, then rebuild the image (`make build`).

## Knowledge Base

*   Place all relevant PDF documents inside the `rag/knowledge_base` directory.
*   The first time the application runs, it will process these PDFs, create embeddings, and save them to the PostgreSQL database.
*   Subsequent runs will use the embeddings stored in the database.

## Makefile Commands

*   `make build`: Builds the Docker image (`rag-app`).
*   `make up`: Runs the application interactively in a Docker container (`rag-container`). Mounts the knowledge base.
*   `make down`: Stops the running container.
*   `make state`: Checks the status of the container.
*   `make logs`: Shows the logs of the running container.
*   `make clean`: Stops and removes the container and the Docker image.
*   `make fclean`: Performs `clean` and prunes unused Docker system resources.
*   `make rebuild`: Cleans, rebuilds the image, and runs the application.
*   `make help`: Displays this list of commands.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
