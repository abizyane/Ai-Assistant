import argparse
from rag.chat import RAGAssistant
from rag.config import DATA_DIR
from rag.database import init_db

def main():
    parser = argparse.ArgumentParser(description="RAG AI Assistant")
    parser.add_argument("--session-id", type=str, help="The session ID to use for the chat history.")
    args = parser.parse_args()

    try:
        init_db()
        assistant = RAGAssistant(data_dir=DATA_DIR, session_id=args.session_id)
        assistant.run_chat_session()
    except Exception as e:
        print(f"Application failed to start: {str(e)}")


if __name__ == "__main__":
    main()
