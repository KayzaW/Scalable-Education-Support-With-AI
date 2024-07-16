from flask import Flask
from app.database import init_db

def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///edu_chatbot_app.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    init_db(app)

    with app.app_context():
        from app import routes
        app.register_blueprint(routes.bp)

    return app