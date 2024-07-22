import os
from flask import Flask
from .database import db

def create_app():
    app = Flask(__name__)

    # Use a fixed SECRET_KEY for development
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default_secret_key')  # Use environment variable or default
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///edu_chatbot.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    app.config['SESSION_COOKIE_NAME'] = 'edu_chatbot_session'


    db.init_app(app)

    with app.app_context():
        db.create_all()

    from .routes import bp as routes_bp
    app.register_blueprint(routes_bp)

    return app
