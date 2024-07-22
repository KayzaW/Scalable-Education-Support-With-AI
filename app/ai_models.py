import os
import re
import faiss
import numpy as np
import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering
)
from .models import db, Corpus
import traceback

# Define paths
base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.abspath(os.path.join(base_dir, '../models'))
dataset_dir = os.path.abspath(os.path.join(base_dir, '../dataset'))
finetune_dir = os.path.abspath(os.path.join(base_dir, '../Fine_tune'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define embedding model name
embedding_model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"

# Initialize Hugging Face pipelines and models

# Text classification pipeline for entailment detection
classifier = pipeline(
    'text-classification',
    model=os.path.join(models_dir, 'roberta-large-mnli'),
    tokenizer=os.path.join(models_dir, 'roberta-large-mnli'),
    device=0 if torch.cuda.is_available() else -1
)

# Text generation pipeline for generating answers
text_generator = pipeline(
    'text-generation',
    model=os.path.join(finetune_dir, 'fine_tuned_qg_model'),
    tokenizer=os.path.join(finetune_dir, 'fine_tuned_qg_model'),
    device=0 if torch.cuda.is_available() else -1
)

# Summarization pipeline for generating feedback summaries
feedback_generator = pipeline(
    'summarization',
    model=os.path.join(models_dir, 'bart-large-cnn'),
    tokenizer=os.path.join(models_dir, 'bart-large-cnn'),
    device=0 if torch.cuda.is_available() else -1
)

# Question answering pipeline for answering questions based on context
question_answerer = pipeline(
    'question-answering',
    model=os.path.join(finetune_dir, 'fine_tuned_qa_model'),
    tokenizer=os.path.join(finetune_dir, 'fine_tuned_qa_model'),
    device=0 if torch.cuda.is_available() else -1
)

# Embedding model for FAISS
embedding_model = AutoModel.from_pretrained(embedding_model_name)
embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)


def get_embedding(text):
    """
    Get the embedding of a text using a transformer model.
    """
    try:
        # Tokenize the input text
        inputs = embedding_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            # Get the output of the embedding model
            outputs = embedding_model(**inputs)
        # Calculate the mean of the last hidden state to get the embedding
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    except Exception as e:
        # Log any exception that occurs
        print(f"Error in get_embedding: {e}")
        traceback.print_exc()
        return np.array([])

def create_faiss_index(embeddings):
    """
    Create a FAISS index for efficient similarity search.
    """
    if embeddings.size == 0:
        raise ValueError("No embeddings provided to create FAISS index.")

    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

def fetch_corpus_embeddings(subject, topic):
    """
    Fetch embeddings from the corpus for a specific subject and topic.
    """
    try:
        # Fetch entries from the database
        entries = Corpus.query.filter_by(subject=subject, topic=topic, content_type='text').all()

        # Debug: Check the entries fetched
        print(f"Fetched entries: {len(entries)}")

        texts = [entry.content for entry in entries]

        if texts:
            embeddings = np.vstack([get_embedding(text) for text in texts])
            print(f"Embeddings shape: {embeddings.shape}, Number of texts: {len(texts)}")
        else:
            embeddings = np.array([])  # Ensure this is always an array
            print("No texts found, returning empty embeddings array.")

        return embeddings, texts

    except Exception as e:
        print(f"Error in fetch_corpus_embeddings: {e}")
        return np.array([]), []

def clean_extracted_text(extracted_text):
    """
    Cleans extracted text by removing unnecessary information.
    """
    cleaned_text = re.sub(r'http[s]?://\S+', '', extracted_text)
    cleaned_text = re.sub(r'This work by .*? licensed under .*? CC .*?\n?', '', cleaned_text, flags=re.I)
    cleaned_text = re.sub(r'Page \d+:', '', cleaned_text)
    cleaned_text = re.sub(r'Turn over for next question', '', cleaned_text, flags=re.I)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def summarize_text(text):
    """
    Summarizes long texts to fit within the token limit of the question-answering model.
    """
    try:
        summary = feedback_generator(text, max_length=200, min_length=80, truncation=True)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error in summarizing text: {e}")
        traceback.print_exc()
        return text  # Fallback to original text if summarization fails

def chunk_text(text, max_length=512):
    """
    Chunk the text into manageable pieces for processing.
    Each chunk will contain up to `max_length` tokens.
    """
    inputs = embedding_tokenizer(text, return_tensors="pt", truncation=False)
    tokenized_text = inputs['input_ids'][0]

    chunks = []
    for i in range(0, len(tokenized_text), max_length):
        chunk = tokenized_text[i:i + max_length]
        chunks.append(chunk)

    return chunks

def answer_student_question_with_faiss(question, subject, topic):
    try:
        # Fetch embeddings and texts from the corpus
        embeddings, texts = fetch_corpus_embeddings(subject, topic)

        if embeddings.size == 0 or not texts:
            return "No relevant content found for the question."

        # Create FAISS index for the embeddings
        index = create_faiss_index(embeddings)

        # Get the embedding for the question (student's input, no "generate" prompt)
        question_embedding = get_embedding(question).squeeze()
        question_embedding = np.expand_dims(question_embedding, axis=0).astype('float32')

        # Perform FAISS search to find relevant texts
        faiss_result = index.search(question_embedding, k=5)

        if isinstance(faiss_result, tuple) and len(faiss_result) == 2:
            distances, indices = faiss_result
            relevant_texts = [texts[idx] for idx in indices[0] if idx < len(texts)]
        else:
            return "Error: FAISS search returned an unexpected result."

        if not relevant_texts:
            return "No relevant content found for the question."

        # Combine all relevant texts and chunk them into manageable pieces
        combined_text = " ".join(relevant_texts)
        chunks = chunk_text(combined_text)

        # Iterate through each chunk and generate answers using the student’s input directly
        best_answer = None
        for chunk in chunks:
            decoded_chunk = embedding_tokenizer.decode(chunk, skip_special_tokens=True)
            response = question_answerer(
                question=question,  # Use student's input here directly
                context=decoded_chunk,
                max_length=512,
                truncation=True
            )

            # Keep the best response
            if len(response['answer']) > 10:
                if not best_answer or len(response['answer']) > len(best_answer):
                    best_answer = response['answer']

        if best_answer:
            return best_answer
        else:
            return "No detailed answer could be found for the question."

    except Exception as e:
        print(f"Error in answer_student_question_with_faiss: {e}")
        traceback.print_exc()
        return "Error generating the answer."


def generate_additional_work(topic, difficulty, student_input=None):
    # Modify to use the student's input for generating a relevant question
    prompt = student_input if student_input else f"Generate a {difficulty} level question on the topic of {topic}."

    # Limit to one question and set max_length for the output to be concise
    additional_work = text_generator(
        prompt,
        max_length=128,  # Adjust max_length as per need for question length
        num_return_sequences=1,  # Ensure only one question is generated
        do_sample=False,  # Avoid sampling to make the output more deterministic
    )

    return additional_work[0]['generated_text']


def generate_feedback_and_summary(detailed_feedback, grade, correct_count, total_questions):
    """
    Combine feedback and provide a summary based on grading results.
    """
    feedback_summary = (
        f"Your grade is {grade}. "
        f"You answered {correct_count} out of {total_questions} questions correctly.\n"
    )

    if grade == "F" or grade == "D":
        feedback_summary += "You need to work more on this topic. Consider reviewing the material or asking for additional exercises."
    elif grade == "C":
        feedback_summary += "Good job, but there's room for improvement. Keep practicing!"
    elif grade == "B":
        feedback_summary += "Well done! You're close to mastering this topic."
    else:
        feedback_summary += "Excellent work! You've mastered this topic."

    # Combine detailed feedback with the summary
    feedback_combined = feedback_summary + "\n\nDetailed Feedback:\n" + "\n".join(detailed_feedback)

    return feedback_combined


def normalize_text(text):
    """
    Normalize the text for comparison: strip spaces, convert to lowercase, remove non-alphanumeric characters.
    """
    return re.sub(r'\W+', '', text.strip().lower())

def grade_submission(extracted_texts, correct_answers, subject, topic):
    correct_count = 0
    detailed_feedback = []  # To store detailed feedback for each question
    total_questions = len(correct_answers)
    total_extracted = len(extracted_texts)

    # Early return if lengths don't match
    if total_extracted != total_questions:
        print(f"Mismatch: {total_extracted} extracted texts but {total_questions} correct answers.")
        return "Error: Mismatch between the number of extracted texts and correct answers.", "F"

    # Set the maximum token length for the model
    max_token_limit = 512  # Maximum for roberta-large model

    for i in range(total_questions):  # Both lengths should now be guaranteed equal
        try:
            extracted_text = extracted_texts[i]
            correct_answer = correct_answers[i]

            question = extracted_text.get('question', '')
            student_answer = extracted_text.get('answer', '')

            # Normalize both the student's answer and the correct answer
            normalized_student_answer = normalize_text(student_answer)
            normalized_correct_answer = normalize_text(correct_answer)

            # Validate presence of question and answer
            if not question or not student_answer:
                detailed_feedback.append(f"Skipped: Question or answer was empty at index {i}.")
                continue

            # Direct comparison of normalized answers
            if normalized_student_answer == normalized_correct_answer:
                correct_count += 1
                detailed_feedback.append(f"Correct: {question} - Your answer: {student_answer}")
                continue

            # Handle numeric comparisons
            try:
                if float(normalized_student_answer) == float(normalized_correct_answer):
                    correct_count += 1
                    detailed_feedback.append(f"Correct: {question} - Your answer: {student_answer}")
                    continue
            except ValueError:
                pass  # Not comparable as numbers, move to classifier check

            # If no direct match, classify using the entailment model
            combined_input = f"Question: {question} [SEP] Answer: {student_answer} [SEP] Expected: {correct_answer}"
            inputs = classifier.tokenizer(
                combined_input,
                return_tensors="pt", truncation=True, padding='max_length', max_length=max_token_limit
            )

            with torch.no_grad():
                outputs = classifier.model(**inputs)

            # Check model outputs for entailment
            prediction = torch.argmax(outputs.logits, dim=1).item()

            if prediction == 2:  # Assuming the index for "entailment" is 2
                correct_count += 1
                detailed_feedback.append(f"Correct: {question} - Your answer: {student_answer}")
            else:
                detailed_feedback.append(f"Incorrect: {question} - Your answer: {student_answer}, Correct answer: {correct_answer}")

        except IndexError as e:
            detailed_feedback.append(f"Error: Index out of range while processing question {i}")
        except Exception as e:
            detailed_feedback.append(f"Error: Unexpected issue with question {i}")

    # Calculate grade based on correct answers
    grade_percentage = (correct_count / total_questions) * 100 if total_questions > 0 else 0
    grade = "F"
    if grade_percentage >= 70:
        grade = "A"
    elif grade_percentage >= 60:
        grade = "B"
    elif grade_percentage >= 50:
        grade = "C"
    elif grade_percentage >= 40:
        grade = "D"

    # Generate feedback summary using the updated function
    feedback_combined = generate_feedback_and_summary(detailed_feedback, grade, correct_count, total_questions)

    return feedback_combined, grade


def get_correct_answers(subject, topic, questions):
    """
    Fetch correct answers for a given subject and topic.
    """
    print(f"Fetching correct answers for subject: '{subject}', topic: '{topic}'")

    # Fetch entries with content_type 'mark scheme'
    entries = Corpus.query.filter_by(subject=subject, topic=topic, content_type="mark scheme").all()

    correct_answers = []
    for question in questions:
        # Find entry with exact match of question in content
        matched_entry = next((entry for entry in entries if question.strip().lower() in entry.content.lower().strip()),
                             None)
        if matched_entry:
            # Assuming that each correct answer follows its respective question
            answer_lines = matched_entry.content.split('\n')
            question_found = False
            for line in answer_lines:
                line = line.strip()
                if not line:
                    continue
                if question_found:
                    # After the question, the next line is considered an answer until we find another question
                    if any(q.strip().lower() in line.lower() for q in questions):
                        break
                    else:
                        correct_answers.append(line)
                        break  # Stop after finding the answer
                elif line.strip().lower() == question.strip().lower():
                    question_found = True
        else:
            print(f"No match found for question: {question}")

    # Ensuring the length matches the number of questions
    if len(correct_answers) != len(questions):
        print(
            f"Warning: Number of correct answers ({len(correct_answers)}) does not match number of questions ({len(questions)}).")

    print(f"Found {len(correct_answers)} correct answers for subject: '{subject}', topic: '{topic}'")
    print(f"Correct answers: {correct_answers}")

    return correct_answers


def add_to_corpus(subject, grade_level, topic, content_type, content):
    """
    Add new content to the corpus database.
    """
    new_entry = Corpus(
        subject=subject,
        grade_level=grade_level,
        topic=topic,
        content_type=content_type,
        content=content
    )
    db.session.add(new_entry)
    db.session.commit()




def extract_texts_from_extracted_text(extracted_text):
    """
    Extract questions and answers from the given text using flexible patterns.
    """
    extracted_texts = []
    current_question = None
    current_answer = None
    question_patterns = [
        r"^\d+\([a-z]\)\s.*",  # Matches patterns like "1(a) Which..." for questions
        r"^\d+[\.\)]\s.*",  # Matches patterns like "1. Question" or "1) Question"
        r"^[a-zA-Z]+\s[\.\)]\s.*",  # Matches patterns like "a. Question" or "A) Question"
        r"^Q\d+\s.*"  # Matches patterns like "Q1 Question"
    ]

    # Combine all patterns into one regex
    combined_pattern = re.compile('|'.join(question_patterns))

    lines = extracted_text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Detect new questions based on patterns
        if combined_pattern.match(line):
            if current_question is not None and current_answer is not None:
                # Only append if we have a valid question and answer
                extracted_texts.append({"question": current_question.strip(), "answer": current_answer.strip()})
            current_question = line  # Start new question
            current_answer = None
        else:
            # Consider any non-matching lines as answers
            if current_answer is None:
                current_answer = line
            else:
                current_answer += " " + line

    # Append the last question-answer pair
    if current_question and current_answer:
        extracted_texts.append({"question": current_question.strip(), "answer": current_answer.strip()})

    print(f"Number of extracted texts: {len(extracted_texts)}")
    print(f"Extracted texts: {extracted_texts}")
    return extracted_texts