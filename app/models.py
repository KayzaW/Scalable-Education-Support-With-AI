from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from .database import db

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'student' or 'teacher'

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Subject(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)

class Topic(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    subject_id = db.Column(db.Integer, db.ForeignKey('subject.id'), nullable=False)

    subject = db.relationship('Subject', backref=db.backref('topics', lazy=True))

class Material(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    subject_id = db.Column(db.Integer, db.ForeignKey('subject.id'), nullable=False)
    topic_id = db.Column(db.Integer, db.ForeignKey('topic.id'), nullable=False)
    file_path = db.Column(db.String(200), nullable=False)
    extracted_text = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    content_type = db.Column(db.String(50), nullable=False)

    subject = db.relationship('Subject', backref=db.backref('materials', lazy=True))
    topic = db.relationship('Topic', backref=db.backref('materials', lazy=True))

    def to_dict(self):
        return {
            "id": self.id,
            "subject": self.subject.name,
            "topic": self.topic.name,
            "file_path": self.file_path,
            "extracted_text": self.extracted_text,
            "created_at": self.created_at,
            "content_type": self.content_type
        }

    def add_to_corpus(self, content_type='text'):
        corpus_entry = Corpus(
            subject=self.subject.name,
            topic=self.topic.name,
            content_type=content_type,
            content=self.extracted_text
        )
        db.session.add(corpus_entry)
        db.session.commit()

class Submission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    subject_id = db.Column(db.Integer, db.ForeignKey('subject.id'), nullable=False)
    topic_id = db.Column(db.Integer, db.ForeignKey('topic.id'), nullable=False)
    file_path = db.Column(db.String(200), nullable=False)
    extracted_text = db.Column(db.Text, nullable=False)
    generated_work = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref=db.backref('submissions', lazy=True))
    subject = db.relationship('Subject', backref=db.backref('submissions', lazy=True))
    topic = db.relationship('Topic', backref=db.backref('submissions', lazy=True))

    def to_dict(self):
        return {
            "id": self.id,
            "subject": self.subject.name,
            "topic": self.topic.name,
            "extracted_text": self.extracted_text,
            "generated_work": self.generated_work,
            "created_at": self.created_at.strftime('%Y-%m-%d %H:%M:%S')
        }

class Corpus(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    subject = db.Column(db.String(50), nullable=False)
    topic = db.Column(db.String(100), nullable=False)
    content_type = db.Column(db.String(50), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "subject": self.subject,
            "topic": self.topic,
            "content_type": self.content_type,
            "content": self.content,
            "created_at": self.created_at
        }
