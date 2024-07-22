import unittest
import numpy as np
from app import create_app
from app.models import db
from app.ai_models import (
    answer_student_question_with_faiss,
    generate_additional_work,
    grade_submission,
    fetch_corpus_embeddings,
    generate_feedback_and_summary
)


class TestAIModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.app = create_app()
        cls.app_context = cls.app.app_context()
        cls.app_context.push()


        db.create_all()

    @classmethod
    def tearDownClass(cls):

        db.session.remove()
        db.drop_all()
        cls.app_context.pop()


    def test_answer_question_valid(self):
        question = "Who is macbeth?"
        subject = "Geography"
        topic = "Countries"

        result = answer_student_question_with_faiss(question, subject, topic)
        self.assertIsNotNone(result, "The function should return an answer.")
        self.assertIsInstance(result, str, "The answer should be a string.")

    def test_answer_question_invalid_subject(self):
        question = "What is the capital of France?"
        subject = "InvalidSubject"
        topic = "Countries"

        result = answer_student_question_with_faiss(question, subject, topic)
        self.assertEqual(result, "No relevant content found for the question.",
                         "The function should handle invalid subjects.")


    def test_generate_question_easy(self):
        topic = "Math"
        difficulty = "easy"

        result = generate_additional_work(topic, difficulty)
        self.assertIsNotNone(result, "The function should return a generated question.")
        self.assertIsInstance(result, str, "The generated question should be a string.")
        self.assertGreater(len(result), 10, "The generated question should not be too short.")

    def test_generate_question_hard(self):
        topic = "Science"
        difficulty = "hard"

        result = generate_additional_work(topic, difficulty)
        self.assertIsNotNone(result, "The function should return a generated question.")
        self.assertGreater(len(result), 10, "The generated question should not be too short.")
        self.assertTrue("question" in result.lower(), "The generated question should reflect the difficulty level.")


    def test_grade_correct_answers(self):
        extracted_texts = [
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "What is the capital of France?", "answer": "Paris"}
        ]
        correct_answers = ["4", "Paris"]
        subject = "Math"
        topic = "Basic Arithmetic"

        feedback, grade = grade_submission(extracted_texts, correct_answers, subject, topic)
        self.assertEqual(grade, "A", "The grade should be 'A' for all correct answers.")
        self.assertIn("Correct", feedback, "Feedback should indicate correct answers.")

    def test_grade_incorrect_answers(self):
        extracted_texts = [
            {"question": "What is 2+2?", "answer": "5"},
            {"question": "What is the capital of France?", "answer": "Berlin"}
        ]
        correct_answers = ["4", "Paris"]
        subject = "Math"
        topic = "Basic Arithmetic"

        feedback, grade = grade_submission(extracted_texts, correct_answers, subject, topic)
        self.assertEqual(grade, "F", "The grade should be 'F' for all incorrect answers.")
        self.assertIn("Incorrect", feedback, "Feedback should indicate incorrect answers.")


    def test_fetch_embeddings_valid(self):
        subject = "English"
        topic = "Macbeth"

        embeddings, texts = fetch_corpus_embeddings(subject, topic)
        self.assertIsInstance(embeddings, np.ndarray, "Embeddings should be returned as a NumPy array.")
        self.assertGreater(len(texts), 0, "Texts should be returned for the given subject and topic.")

    def test_fetch_embeddings_invalid(self):
        subject = "InvalidSubject"
        topic = "InvalidTopic"

        embeddings, texts = fetch_corpus_embeddings(subject, topic)
        self.assertEqual(embeddings.size, 0, "Embeddings should be empty for invalid subject and topic.")
        self.assertEqual(len(texts), 0, "No texts should be returned for an invalid subject and topic.")


    def test_generate_feedback(self):
        detailed_feedback = ["Correct: What is 2+2?", "Incorrect: What is the capital of France?"]
        correct_count = 1
        total_questions = 2

        feedback = generate_feedback_and_summary(detailed_feedback, correct_count, total_questions)
        self.assertIn("Your grade is", feedback, "Feedback should include the grade.")
        self.assertIn("Correct", feedback, "Feedback should include detailed information on correct answers.")
        self.assertIn("Incorrect", feedback, "Feedback should include detailed information on incorrect answers.")

    def test_generate_feedback_all_correct(self):
        detailed_feedback = ["Correct: What is 2+2?", "Correct: What is the capital of France?"]
        correct_count = 2
        total_questions = 2

        feedback = generate_feedback_and_summary(detailed_feedback, correct_count, total_questions)
        self.assertIn("Your grade is A", feedback, "The grade should be 'A' for all correct answers.")


if __name__ == '__main__':
    unittest.main()
