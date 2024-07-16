from flask import Blueprint, request, jsonify
from app.models import db, User, Assignment, PerformanceLog, ChatLog
from app.ai_models import grade_assignment, generate_exercise, generate_feedback, identify_weaknesses, train_rag_model
from app.ocr import extract_text_from_image
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import pandas as pd

bp = Blueprint('routes', __name__)


@bp.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    user = User.query.filter_by(username=username).first()
    if user and check_password_hash(user.password_hash, password):
        return jsonify({"message": "Login successful", "username": username})
    return jsonify({"message": "Invalid credentials"}), 401


@bp.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    email = data.get('email')
    password_hash = generate_password_hash(password)
    new_user = User(username=username, password_hash=password_hash, email=email, role='student')
    db.session.add(new_user)
    db.session.commit()
    return jsonify({"message": "User registered successfully"})


@bp.route('/grade', methods=['POST'])
def grade():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Extract text from image
        assignment_content = extract_text_from_image(filepath)

        # Process the extracted text
        result = grade_assignment(assignment_content)
        return jsonify(result)
    return jsonify({"message": "Invalid file type"}), 400


@bp.route('/generate_exercise', methods=['POST'])
def generate_exercise_route():
    data = request.json
    prompt = data.get('prompt')
    result = generate_exercise(prompt)
    return jsonify(result)


@bp.route('/feedback', methods=['POST'])
def feedback_route():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Extract text from image
        assignment_content = extract_text_from_image(filepath)

        # Process the extracted text
        result = generate_feedback(assignment_content)
        return jsonify(result)
    return jsonify({"message": "Invalid file type"}), 400


@bp.route('/identify_weaknesses', methods=['POST'])
def identify_weaknesses_route():
    file = request.files['file']
    question = request.form['question']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Extract text from image
        assignment_content = extract_text_from_image(filepath)

        # Process the extracted text
        result = identify_weaknesses(assignment_content, question)
        return jsonify(result)
    return jsonify({"message": "Invalid file type"}), 400


@bp.route('/performance/<student_id>', methods=['GET'])
def get_performance(student_id):
    assignments = Assignment.query.filter_by(student_id=student_id).all()
    performance_data = []
    for assignment in assignments:
        performance_logs = PerformanceLog.query.filter_by(assignment_id=assignment.id).all()
        for log in performance_logs:
            performance_data.append({
                "assignment_id": assignment.id,
                "performance_data": log.performance_data,
                "timestamp": log.timestamp
            })
    insights = analyze_performance(performance_data)
    return jsonify(insights)


def analyze_performance(performance_data):
    df = pd.DataFrame(performance_data)
    overall_performance = "A"
    individual_progress = []
    specific_challenges = []
    for index, row in df.iterrows():
        pass
    return {
        "overall_performance": overall_performance,
        "individual_progress": individual_progress,
        "specific_challenges": specific_challenges
    }


@bp.route('/chat_logs/<user_id>', methods=['GET'])
def get_chat_logs(user_id):
    chat_logs = ChatLog.query.filter_by(user_id=user_id).all()
    logs = [{"message": log.message, "timestamp": log.timestamp} for log in chat_logs]
    return jsonify(logs)


@bp.route('/train_model', methods=['POST'])
def train_model():
    data = request.json
    dataset_path = data.get('dataset_path')
    if not os.path.exists(dataset_path):
        return jsonify({"message": "Dataset not found"}), 400
    train_rag_model(dataset_path)
    return jsonify({"message": "Model training initiated"}), 200


def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
