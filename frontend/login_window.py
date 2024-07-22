from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
import requests

class LoginWindow(QWidget):
    def __init__(self, parent=None):
        super(LoginWindow, self).__init__(parent)
        self.parent = parent
        self.session = requests.Session()
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        title = QLabel("Welcome to EduChatBot")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 24))
        main_layout.addWidget(title)


        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Enter your username")
        self.username_input.setFont(QFont("Arial", 14))
        self.username_input.setFixedHeight(40)
        main_layout.addWidget(self.username_input)


        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Enter your password")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setFont(QFont("Arial", 14))
        self.password_input.setFixedHeight(40)
        main_layout.addWidget(self.password_input)

        self.login_button = QPushButton("Login")
        self.login_button.setFont(QFont("Arial", 14))
        self.login_button.clicked.connect(self.handle_login)
        main_layout.addWidget(self.login_button)

        self.setLayout(main_layout)
        self.setWindowTitle("EduChatBot Login")
        self.setGeometry(300, 300, 400, 300)

    def handle_login(self):
        username = self.username_input.text()
        password = self.password_input.text()

        try:

            response = self.session.post('http://localhost:5000/api/login',
                                         json={'username': username, 'password': password})

            if response.status_code == 200:
                user_data = response.json()
                user_role = user_data.get('role')
                user_id = user_data.get('user_id')
                print("Cookies after login:", self.session.cookies)
                if user_role == 'student':
                    self.parent.show_student_dashboard(user_id, self.session)
                elif user_role == 'teacher':
                    self.parent.show_teacher_dashboard(user_id, self.session)
            else:
                QMessageBox.warning(self, "Error", "Invalid credentials")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")
