from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox, QFormLayout, QHBoxLayout, QFrame, QSizePolicy
from PyQt5.QtGui import QFont, QIcon, QPixmap
from PyQt5.QtCore import Qt
import requests

class LoginWindow(QWidget):
    def __init__(self, parent=None):
        super(LoginWindow, self).__init__(parent)
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)

        # Title
        title = QLabel("Welcome to EduChatBot")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 24))
        main_layout.addWidget(title)

        # Logo
        logo = QLabel(self)
        pixmap = QPixmap("path/to/school_logo.png")  # Ensure you have a school logo image
        logo.setPixmap(pixmap)
        logo.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(logo)

        # Form layout
        form_layout = QVBoxLayout()
        form_layout.setAlignment(Qt.AlignCenter)

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Enter your username")
        self.username_input.setFont(QFont("Arial", 14))
        self.username_input.setFixedHeight(40)
        self.username_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Enter your password")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setFont(QFont("Arial", 14))
        self.password_input.setFixedHeight(40)
        self.password_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        form_layout.addWidget(self.username_input)
        form_layout.addWidget(self.password_input)

        form_frame = QFrame()
        form_frame.setLayout(form_layout)
        form_frame.setFrameShape(QFrame.StyledPanel)
        form_frame.setFrameShadow(QFrame.Raised)
        main_layout.addWidget(form_frame)

        # Buttons layout
        buttons_layout = QHBoxLayout()
        buttons_layout.setAlignment(Qt.AlignCenter)

        self.login_button = QPushButton("Login")
        self.login_button.setFont(QFont("Arial", 14))
        self.login_button.setFixedHeight(40)
        self.login_button.setFixedWidth(100)
        self.login_button.clicked.connect(self.handle_login)
        buttons_layout.addWidget(self.login_button)

        main_layout.addLayout(buttons_layout)

        # Set the main layout
        self.setLayout(main_layout)
        self.setWindowTitle("EduChatBot Login")
        self.setGeometry(300, 300, 400, 500)

    def handle_login(self):
        username = self.username_input.text()
        password = self.password_input.text()
        response = requests.post('http://localhost:5000/login', json={'username': username, 'password': password})

        if response.status_code == 200:
            user_role = response.json().get('role')
            if user_role == 'student':
                self.parent.show_student_dashboard()
            elif user_role == 'teacher':
                self.parent.show_teacher_page()
        else:
            QMessageBox.warning(self, "Error", "Invalid credentials")
