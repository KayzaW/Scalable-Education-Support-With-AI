import os
import time
from app import create_app  # Make sure you have this in your Flask project
from app.ai_models import answer_student_question_with_faiss, generate_additional_work

# Initialize Flask app
app = create_app()

# Setting up the Flask application context
with app.app_context():
    def test_question_answering():
        """
        Function to test the Question Answering model with FAISS
        """
        question = "How does Macbeth become king of Scotland?"
        subject = "English"
        topic = "Macbeth"

        try:
            start_time = time.time()
            result = answer_student_question_with_faiss(question, subject, topic)
            end_time = time.time()

            print(f"Question Answering Model Inference Time: {end_time - start_time:.4f} seconds")
            print(f"Question Answering Output: {result}")

        except Exception as e:
            print(f"Error during Question Answering: {e}")


    def test_question_generation():
        """
        Function to test the Question Generation model
        """
        topic = "Math"
        difficulty = "easy"

        try:
            start_time = time.time()
            result = generate_additional_work(topic, difficulty)
            end_time = time.time()

            print(f"Question Generation Model Inference Time: {end_time - start_time:.4f} seconds")
            print(f"Question Generation Output: {result}")

        except Exception as e:
            print(f"Error during Question Generation: {e}")


    def run_inference_tests():
        """
        Run inference tests for both Question Answering and Question Generation models.
        """
        print("Testing Question Answering Model...")
        test_question_answering()

        print("\nTesting Question Generation Model...")
        test_question_generation()

        print("\n--- Inference Results Summary ---")
        # Add additional summary or metrics if needed


    if __name__ == '__main__':
        run_inference_tests()
