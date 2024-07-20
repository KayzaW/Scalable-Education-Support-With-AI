from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox, QListWidget, QListWidgetItem, QTextEdit, QLineEdit, QInputDialog
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
import requests

class StudentDashboard(QWidget):
    def __init__(self, parent=None):
        super(StudentDashboard, self).__init__(parent)
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        # Main layout
        main_layout = QHBoxLayout()

        # Left menu layout
        menu_layout = QVBoxLayout()
        menu_layout.setAlignment(Qt.AlignTop)

        # Title
        title = QLabel("EduChatBot")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Comic Sans MS", 20))
        title.setStyleSheet("color: #4CAF50;")
        menu_layout.addWidget(title)

        # List of past pieces of work
        self.past_work_list = QListWidget()
        self.past_work_list.setFont(QFont("Comic Sans MS", 12))
        self.load_past_work()
        menu_layout.addWidget(self.past_work_list)

        # Add new work button
        self.add_work_button = QPushButton("Add New Work")
        self.add_work_button.setFont(QFont("Comic Sans MS", 14))
        self.add_work_button.setFixedHeight(40)
        self.add_work_button.setStyleSheet("background-color: #2196F3; color: white; border-radius: 10px;")
        self.add_work_button.clicked.connect(self.add_new_work)
        menu_layout.addWidget(self.add_work_button)

        # Upload and grade new work button
        self.upload_button = QPushButton("Upload and Grade Work")
        self.upload_button.setFont(QFont("Comic Sans MS", 14))
        self.upload_button.setFixedHeight(40)
        self.upload_button.setStyleSheet("background-color: #4CAF50; color: white; border-radius: 10px;")
        self.upload_button.clicked.connect(self.upload_and_grade_work)
        menu_layout.addWidget(self.upload_button)

        main_layout.addLayout(menu_layout)

        # Chat layout
        chat_layout = QVBoxLayout()
        chat_layout.setAlignment(Qt.AlignTop)

        # Chat area
        self.chat_area = QTextEdit()
        self.chat_area.setFont(QFont("Comic Sans MS", 14))
        self.chat_area.setReadOnly(True)
        chat_layout.addWidget(self.chat_area)

        # Input area
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setFont(QFont("Comic Sans MS", 14))
        self.input_field.setPlaceholderText("Type a message...")
        self.input_field.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.input_field)

        self.send_button = QPushButton("Send")
        self.send_button.setFont(QFont("Comic Sans MS", 14))
        self.send_button.setStyleSheet("background-color: #4CAF50; color: white; border-radius: 10px;")
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_button)

        chat_layout.addLayout(input_layout)

        main_layout.addLayout(chat_layout)

        self.setLayout(main_layout)

    def load_past_work(self):
        # Dummy data for past work, in practice you should fetch this from your backend or database
        past_works = ["Math Homework 1", "Science Project", "English Essay"]
        for work in past_works:
            item = QListWidgetItem(work)
            self.past_work_list.addItem(item)

    def add_new_work(self):
        # Logic to add a new piece of work
        new_work_name, ok = QInputDialog.getText(self, "New Work", "Enter the name of the new work:")
        if ok and new_work_name:
            item = QListWidgetItem(new_work_name)
            self.past_work_list.addItem(item)
            # Optionally save this new work to backend or database

    def upload_and_grade_work(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Upload Homework", "", "Images (*.png *.jpg *.jpeg)", options=options)
        if file_path:
            files = {'file': open(file_path, 'rb')}
            response = requests.post('http://localhost:5000/grade', files=files)
            if response.status_code == 200:
                result = response.json()
                self.show_results(result)
            else:
                QMessageBox.warning(self, "Error", "Failed to upload and process homework")

    def show_results(self, result):
        feedback = result.get('feedback', 'No feedback available')
        grade = result.get('grade', 'No grade available')
        additional_work = result.get('additional_work', 'No additional work available')

        self.chat_area.append(f"Feedback: {feedback}")
        self.chat_area.append(f"Grade: {grade}")
        self.chat_area.append(f"Additional Work: {additional_work}")

    def send_message(self):
        message = self.input_field.text()
        if message:
            self.chat_area.append(f"You: {message}")
            self.input_field.clear()
            # Logic to handle sending message to backend or further processing
