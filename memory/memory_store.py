import sqlite3
import json
import os
from datetime import datetime

class MemoryStore:
    def __init__(self, db_path="data/math_mentor_memory.sqlite"):
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    original_input TEXT,
                    parsed_question_json TEXT,
                    retrieved_context TEXT,
                    final_solution TEXT,
                    verifier_outcome TEXT,
                    user_feedback TEXT
                )
            ''')
            conn.commit()

    def save_interaction(self, session_id: str, original_input: str, parsed_question: dict,
                         retrieved_context: str, final_solution: str, verifier_outcome: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO interactions (
                    session_id, original_input, parsed_question_json, 
                    retrieved_context, final_solution, verifier_outcome
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                session_id,
                original_input,
                json.dumps(parsed_question),
                retrieved_context,
                final_solution,
                verifier_outcome
            ))
            conn.commit()
            return cursor.lastrowid

    def update_feedback(self, interaction_id: int, is_correct: bool, comment: str):
        feedback_data = json.dumps({"is_correct": is_correct, "comment": comment})
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE interactions 
                SET user_feedback = ? 
                WHERE id = ?
            ''', (feedback_data, interaction_id))
            conn.commit()

    def get_history(self, session_id: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM interactions WHERE session_id = ? ORDER BY timestamp DESC', (session_id,))
            return [dict(row) for row in cursor.fetchall()]
