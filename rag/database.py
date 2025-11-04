import psycopg2
from rag.config import DATABASE_URL
from typing import List, Tuple

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except psycopg2.OperationalError as e:
        print(f"Error connecting to the database: {e}")
        return None

def init_db():
    """Initializes the database by creating the vector extension and the necessary tables."""
    conn = get_db_connection()
    if conn is None:
        return

    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding VECTOR(384),
                    metadata JSONB
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    assistant_message TEXT NOT NULL
                );
            """)
        conn.commit()
    except Exception as e:
        print(f"Error initializing the database: {e}")
    finally:
        if conn:
            conn.close()

def save_chat_history(session_id: str, user_message: str, assistant_message: str):
    """Saves a chat interaction to the database."""
    conn = get_db_connection()
    if conn is None:
        return

    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chat_history (session_id, user_message, assistant_message) VALUES (%s, %s, %s)",
                (session_id, user_message, assistant_message)
            )
        conn.commit()
    except Exception as e:
        print(f"Error saving chat history: {e}")
    finally:
        if conn:
            conn.close()

def load_chat_history(session_id: str) -> List[Tuple[str, str]]:
    """Loads the chat history for a given session from the database."""
    conn = get_db_connection()
    if conn is None:
        return []

    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_message, assistant_message FROM chat_history WHERE session_id = %s ORDER BY id",
                (session_id,)
            )
            history = cur.fetchall()
            return [(row[0], row[1]) for row in history]
    except Exception as e:
        print(f"Error loading chat history: {e}")
        return []
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    init_db()
