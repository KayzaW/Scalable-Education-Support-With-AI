from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox
import requests

class LoginWindow(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        self.layout = QVBoxLayout()

        self.label = QLabel("Login")
        self.layout.addWidget(self.label)

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Username")
        self.layout.addWidget(self.username_input)

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Password")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.layout.addWidget(self.password_input)

        self.login_button = QPushButton("Login")
        self.login_button.clicked.connect(self.login)
        self.layout.addWidget(self.login_button)

        self.setLayout(self.layout)

    def login(self):
        username = self.username_input.text()
        password = self.password_input.text()
        response = requests.post('http://localhost:5000/login', json={'username': username, 'password': password})
        if response.status_code == 200:
            data = response.json()
            if data['role'] == 'student':
                self.parent.show_student_dashboard()
            else:
                self.parent.show_teacher_dashboard()
        else:
            QMessageBox.critical(self, "Error", "Invalid credentials")
