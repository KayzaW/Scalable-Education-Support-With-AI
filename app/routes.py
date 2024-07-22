from flask import Blueprint, request, jsonify, session
from .models import db, User, Subject, Topic, Material, Submission, Corpus
from .ocr import extract_text_from_file, extract_text_from_image, extract_text_from_pdf, extract_text_from_docx
from .ai_models import answer_student_question_with_faiss, generate_feedback_and_summary, grade_submission, add_to_corpus, get_correct_answers, extract_texts_from_extracted_text, fetch_corpus_embeddings, generate_additional_work, get_embedding, clean_extracted_text, create_faiss_index
from werkzeug.utils import secure_filename
import os
from .utils import login_required
import numpy as np
import traceback
bp = Blueprint('routes', __name__)

uploads_dir = 'uploads'

@bp.route('/api/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    role = data.get('role')  # 'student' or 'teacher'

    if User.query.filter_by(username=username).first():
        return jsonify({"message": "Username already exists"}), 400

    new_user = User(username=username, role=role)
    new_user.set_password(password)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"message": "Registration successful"}), 200

@bp.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.json
        username = data.get('username')
        password = data.get('password')

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            # Set session variables
            session['user_id'] = user.id
            session['role'] = user.role
            print(f"User {user.id} logged in with role {user.role}")  # Debug output

            # Explicitly print session for debugging
            print("Session after login:", dict(session))

            return jsonify({
                "message": "Login successful",
                "user_id": user.id,
                "role": user.role
            }), 200

        return jsonify({"message": "Invalid credentials"}), 401
    except Exception as e:
        print(f"Error in login route: {e}")
        return jsonify({"message": "Server error"}), 500

@bp.route('/api/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)
    session.pop('role', None)
    return jsonify({"message": "Logout successful"}), 200

@bp.route('/api/subjects', methods=['GET'])
def get_subjects():
    subjects = Subject.query.all()
    return jsonify([{'id': subject.id, 'name': subject.name} for subject in subjects]), 200

@bp.route('/api/add_subject', methods=['POST'])
def add_subject():
    data = request.json
    subject_name = data.get('subject')

    if not subject_name:
        return jsonify({"message": "Invalid input"}), 400

    new_subject = Subject(name=subject_name)
    db.session.add(new_subject)
    db.session.commit()

    return jsonify({"message": "Subject added successfully"}), 200

@bp.route('/api/topics/<int:subject_id>', methods=['GET'])
def get_topics(subject_id):
    topics = Topic.query.filter_by(subject_id=subject_id).all()
    return jsonify([{'id': topic.id, 'name': topic.name} for topic in topics]), 200

@bp.route('/api/add_topic', methods=['POST'])
def add_topic():
    data = request.json
    subject_id = data.get('subject_id')
    topic_name = data.get('topic')

    subject = Subject.query.get(subject_id)
    if not subject:
        return jsonify({"message": "Subject not found"}), 404

    new_topic = Topic(name=topic_name, subject_id=subject.id)
    db.session.add(new_topic)
    db.session.commit()

    return jsonify({"message": "Topic added successfully"}), 200

@bp.route('/api/materials/<int:subject_id>', methods=['GET'])
def get_materials(subject_id):
    materials = Material.query.filter_by(subject_id=subject_id).all()
    return jsonify([material.to_dict() for material in materials]), 200

@bp.route('/api/upload_material', methods=['POST'])
def upload_material():
    try:
        subject_id = request.form.get('subject_id')
        topic_id = request.form.get('topic_id')
        content_type = request.form.get('content_type')
        file = request.files.get('file')

        if not file or not subject_id or not topic_id or not content_type:
            return jsonify({"message": "Invalid input"}), 400

        os.makedirs(uploads_dir, exist_ok=True)

        filename = secure_filename(file.filename)
        file_path = os.path.join(uploads_dir, filename)
        file.save(file_path)

        extracted_text = ""
        if filename.endswith('.pdf'):
            extracted_text = extract_text_from_pdf(file_path)
        elif filename.endswith('.docx'):
            extracted_text = extract_text_from_docx(file_path)
        elif filename.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            extracted_text = extract_text_from_image(file_path)
        else:
            return jsonify({"message": "Unsupported file format"}), 400

        new_material = Material(
            subject_id=subject_id,
            topic_id=topic_id,
            content_type=content_type,
            file_path=file_path,
            extracted_text=extracted_text
        )
        db.session.add(new_material)
        db.session.commit()

        # Add extracted content to corpus
        new_material.add_to_corpus(content_type=content_type)

        return jsonify({"message": "Material uploaded successfully"}), 201

    except Exception as e:
        print(f"Error in upload_material: {e}")
        return jsonify({"message": "Server error"}), 500

@bp.route('/api/delete_material/<int:material_id>', methods=['DELETE'])
def delete_material(material_id):
    material = Material.query.get(material_id)
    if not material:
        return jsonify({"message": "Material not found"}), 404

    db.session.delete(material)
    db.session.commit()

    return jsonify({"message": "Material deleted successfully"}), 200

@bp.route('/api/all_submissions', methods=['GET'])
def get_all_submissions():
    all_submissions = Submission.query.all()
    submissions_by_student = {}

    for submission in all_submissions:
        student = submission.user.username
        if student not in submissions_by_student:
            submissions_by_student[student] = []
        submissions_by_student[student].append({
            "subject": submission.subject.name,
            "topic": submission.topic.name,
            "created_at": submission.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            "grade": submission.generated_work,
            "feedback": submission.extracted_text
        })

    return jsonify(submissions_by_student), 200

@bp.route('/api/upload_work', methods=['POST'])
@login_required
def upload_work():
    try:
        subject_id = request.form.get('subject_id')
        topic_id = request.form.get('topic_id')
        file = request.files.get('file')

        if not file or not subject_id or not topic_id:
            print("Invalid input: missing file, subject_id, or topic_id.")
            return jsonify({"message": "Invalid input"}), 400

        subject = Subject.query.get(subject_id)
        topic = Topic.query.get(topic_id)
        if not subject or not topic:
            print(f"Subject or Topic not found. subject_id: {subject_id}, topic_id: {topic_id}")
            return jsonify({"message": "Subject or Topic not found"}), 400

        subject_name = subject.name
        topic_name = topic.name
        print(f"Found subject: {subject_name}, topic: {topic_name}")

        filename = secure_filename(file.filename)
        file_path = os.path.join(uploads_dir, filename)
        file.save(file_path)
        print(f"File saved at {file_path}")

        file_extension = os.path.splitext(filename)[1].lower()
        print(f"Processing file type: {file_extension}")

        extracted_text = extract_text_from_file(file_path, file_extension)
        if not extracted_text:
            print(f"Extraction failed for file: {file_path} with type {file_extension}")
            return jsonify({"message": "Failed to extract text from the file"}), 400

        extracted_texts = extract_texts_from_extracted_text(extracted_text)
        print(f"Number of extracted texts: {len(extracted_texts)}")

        if not extracted_texts:
            print("No questions and answers were extracted from the text.")
            return jsonify({"message": "Failed to extract questions and answers"}), 400

        extracted_questions = [text['question'] for text in extracted_texts]
        correct_answers = get_correct_answers(subject_name, topic_name, extracted_questions)
        print(f"Number of correct answers fetched: {len(correct_answers)}")

        if len(extracted_texts) != len(correct_answers):
            print(f"Error: Length of extracted texts ({len(extracted_texts)}) and correct answers ({len(correct_answers)}) do not match.")
            return jsonify({"message": "Error processing submission: mismatched lengths."}), 500

        for i, (extracted, correct) in enumerate(zip(extracted_texts, correct_answers)):
            print(f"Comparing extracted text {i}: {extracted['answer']} with correct answer: {correct}")

        feedback, grade = grade_submission(extracted_texts, correct_answers, subject_name, topic_name)

        print(f"Feedback: {feedback}")
        print(f"Grade: {grade}")

        if feedback is None or grade is None:
            print("Grading function returned None for feedback or grade.")
            return jsonify({"message": "Error in grading function"}), 400

        submission = Submission(
            user_id=session['user_id'],
            subject_id=subject_id,
            topic_id=topic_id,
            file_path=file_path,
            extracted_text=extracted_text,
            generated_work=grade
        )
        db.session.add(submission)
        db.session.commit()

        return jsonify({
            "message": "Work uploaded successfully",
            "grade": grade,
            "feedback": feedback
        }), 200

    except Exception as e:
        print(f"Error in upload_work: {e}")
        return jsonify({"message": "Server error"}), 500

@bp.route('/api/student_query', methods=['POST'])
def student_query():
    try:
        # Ensure request is JSON
        if request.content_type != 'application/json':
            return jsonify({"message": "Unsupported Media Type: Expected application/json"}), 415

        # Parse the JSON payload
        data = request.get_json()
        question = data.get('question')
        subject_id = data.get('subject_id')
        topic_id = data.get('topic_id')

        # Validate input
        if not question or not subject_id or not topic_id:
            return jsonify({"message": "Invalid input"}), 400

        # Lookup subject and topic in the database
        subject = Subject.query.get(subject_id)
        topic = Topic.query.get(topic_id)
        if not subject or not topic:
            return jsonify({"message": "Subject or Topic not found"}), 404

        subject_name = subject.name
        topic_name = topic.name

        # Fetch embeddings and texts
        embeddings, texts = fetch_corpus_embeddings(subject_name, topic_name)
        if embeddings.size == 0:
            return jsonify({"message": "No relevant content found for the question."}), 404

        # Create FAISS index and search for the top 3 most relevant texts
        index = create_faiss_index(embeddings)
        question_embedding = get_embedding(question[:512])  # Truncate the question input to avoid overflow
        if question_embedding.size == 0:
            return jsonify({"message": "Error generating question embedding"}), 500

        question_embedding = question_embedding.squeeze()
        question_embedding = np.expand_dims(question_embedding, axis=0).astype('float32')

        faiss_result = index.search(question_embedding, k=3)

        if isinstance(faiss_result, tuple) and len(faiss_result) == 2:
            distances, indices = faiss_result
        else:
            return jsonify({"message": "Error performing FAISS search"}), 500

        relevant_texts = [clean_extracted_text(texts[idx]) for idx in indices[0] if idx < len(texts)]
        if not relevant_texts:
            return jsonify({"message": "No relevant content found for the question."}), 404

        # Check if the query is for generating a new question or answering an existing one
        if "generate" in question.lower():
            response = generate_additional_work(topic_name, "easy")
        else:
            # Use FAISS to find relevant contexts and provide a meaningful answer
            response = answer_student_question_with_faiss(question, subject_name, topic_name)

        final_response = {
            "question": question,
            "answer": response
        }

        return jsonify(final_response), 200

    except Exception as e:
        print(f"Error in student_query: {e}")
        traceback.print_exc()
        return jsonify({"message": "Server error"}), 500


@bp.route('/api/submissions/<int:user_id>', methods=['GET'])
def get_user_submissions(user_id):
    try:
        submissions = Submission.query.filter_by(user_id=user_id).all()
        submissions_list = [{
            "subject": submission.subject.name,
            "topic": submission.topic.name,
            "created_at": submission.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            "grade": submission.generated_work,
            "feedback": submission.extracted_text
        } for submission in submissions]

        return jsonify(submissions_list), 200

    except Exception as e:
        print(f"Error in get_user_submissions: {e}")
        return jsonify({"message": "Server error"}), 500
