from app import create_app
from app.models import User, Subject, Topic
from werkzeug.security import generate_password_hash
from app.database import db

app = create_app()

with app.app_context():
    # Create users
    teacher = User(username="teacher", role="teacher")
    teacher.set_password("123456")
    db.session.add(teacher)

    student = User(username="student", role="student")
    student.set_password("123456")
    db.session.add(student)

    # Create subjects
    science = Subject(name="Science")
    english = Subject(name="English")
    math = Subject(name="Mathematics")
    db.session.add(science)
    db.session.add(english)
    db.session.add(math)


    db.session.commit()  # Commit the subjects first to get their IDs

    # Create topics under English subject
    macbeth = Topic(name="Macbeth", subject_id=english.id)
    poetry = Topic(name="Poetry", subject_id=english.id)
    db.session.add(macbeth)
    db.session.add(poetry)

    # Create topics under Mathematics subject
    algebra = Topic(name="Foundation", subject_id=math.id)
    geometry = Topic(name="Geometry", subject_id=math.id)
    db.session.add(algebra)
    db.session.add(geometry)

    # Create topics under Science subject
    physics = Topic(name="Physics", subject_id=science.id)
    chemistry = Topic(name="Chemistry", subject_id=science.id)
    db.session.add(physics)
    db.session.add(chemistry)

    db.session.commit()

print("Database populated successfully!")
