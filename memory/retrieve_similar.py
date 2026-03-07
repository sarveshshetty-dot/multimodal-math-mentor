import sqlite3
import json

def retrieve_similar_problems(topic: str, db_path="data/math_mentor_memory.sqlite", limit: int = 3) -> list:
    """
    Retrieves previously solved problems from the memory store that match the given topic 
    and were marked as correct by the user. This helps the solver agent reuse solution patterns.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            # Simple retrieval strategy: look for matching topics in the parsed question JSON 
            # where the user feedback was positively marked.
            cursor.execute('''
                SELECT original_input, final_solution, parsed_question_json
                FROM interactions 
                WHERE user_feedback LIKE '%"is_correct": true%'
                ORDER BY timestamp DESC
            ''')
            rows = cursor.fetchall()
            
            similar_problems = []
            for row in rows:
                try:
                    parsed_dict = json.loads(row['parsed_question_json'])
                    if parsed_dict.get('topic', '').lower() == topic.lower():
                        similar_problems.append({
                            "problem": row['original_input'],
                            "solution": row['final_solution']
                        })
                        if len(similar_problems) >= limit:
                            break
                except json.JSONDecodeError:
                    continue
            return similar_problems
    except sqlite3.OperationalError:
        # DB might not exist yet
        return []
