from PyQt5.QtWidgets import QWidget, QListWidgetItem, QVBoxLayout, QLabel, QPushButton, QListWidget, QFileDialog, QMessageBox, QLineEdit, QTextEdit, QComboBox
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
import requests

class StudentDashboard(QWidget):
    def __init__(self, parent=None, user_id=None, session=None):
        super(StudentDashboard, self).__init__(parent)
        self.parent = parent
        self.user_id = user_id
        self.session = session
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        title = QLabel("EduChatBot - Student Dashboard")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Comic Sans MS", 20))
        title.setStyleSheet("color: #4CAF50;")
        main_layout.addWidget(title)

        self.logout_button = QPushButton("Logout")
        self.logout_button.setFont(QFont("Comic Sans MS", 12))
        self.logout_button.setFixedHeight(40)
        self.logout_button.setStyleSheet("background-color: #FF5733; color: white; border-radius: 10px;")
        self.logout_button.clicked.connect(self.handle_logout)
        self.logout_button.setFixedWidth(100)
        main_layout.addWidget(self.logout_button, alignment=Qt.AlignRight)

        self.past_work_list = QListWidget()
        self.past_work_list.setFont(QFont("Comic Sans MS", 12))
        self.load_past_work()
        main_layout.addWidget(self.past_work_list)

        self.upload_and_grade_button = QPushButton("Upload and Grade New Work")
        self.upload_and_grade_button.setFont(QFont("Comic Sans MS", 14))
        self.upload_and_grade_button.setFixedHeight(40)
        self.upload_and_grade_button.setStyleSheet("background-color: #4CAF50; color: white; border-radius: 10px;")
        self.upload_and_grade_button.clicked.connect(self.upload_and_grade_work)
        main_layout.addWidget(self.upload_and_grade_button)

        subject_label = QLabel("Select Subject")
        subject_label.setFont(QFont("Comic Sans MS", 14))
        main_layout.addWidget(subject_label)

        self.subject_input = QComboBox()
        self.load_subjects()
        self.subject_input.currentIndexChanged.connect(self.load_topics)
        main_layout.addWidget(self.subject_input)

        topic_label = QLabel("Select Topic")
        topic_label.setFont(QFont("Comic Sans MS", 14))
        main_layout.addWidget(topic_label)

        self.topic_input = QComboBox()
        main_layout.addWidget(self.topic_input)

        self.query_input = QLineEdit()
        self.query_input.setFont(QFont("Comic Sans MS", 14))
        self.query_input.setPlaceholderText("Ask a question...")
        main_layout.addWidget(self.query_input)

        self.query_button = QPushButton("Ask")
        self.query_button.setFont(QFont("Comic Sans MS", 14))
        self.query_button.setStyleSheet("background-color: #4CAF50; color: white; border-radius: 10px;")
        self.query_button.clicked.connect(self.ask_question)
        main_layout.addWidget(self.query_button)

        chat_layout = QVBoxLayout()
        chat_layout.setAlignment(Qt.AlignTop)

        self.chat_area = QTextEdit()
        self.chat_area.setFont(QFont("Comic Sans MS", 14))
        self.chat_area.setReadOnly(True)
        self.chat_area.setFixedHeight(300)  # Increase the size of the chatbox
        chat_layout.addWidget(self.chat_area)

        self.help_button = QPushButton("Help")
        self.help_button.setFont(QFont("Comic Sans MS", 14))
        self.help_button.setStyleSheet("background-color: #2196F3; color: white; border-radius: 10px;")
        self.help_button.clicked.connect(self.show_help)
        chat_layout.addWidget(self.help_button)

        main_layout.addLayout(chat_layout)
        self.setLayout(main_layout)

    def handle_logout(self):
        try:
            response = self.session.post('http://localhost:5000/api/logout')
            if response.status_code == 200:
                QMessageBox.information(self, "Logged Out", "You have been logged out.")
                self.parent.stack.setCurrentWidget(self.parent.login_window)
            else:
                QMessageBox.warning(self, "Error", "Logout failed.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def load_past_work(self):
        try:
            response = self.session.get(f'http://localhost:5000/api/submissions/{self.user_id}')
            if response.status_code == 200:
                past_works = response.json()
                self.past_work_list.clear()
                for work in past_works:
                    item = QListWidgetItem(f"{work['subject']} - {work['topic']} - {work['created_at']}")
                    self.past_work_list.addItem(item)
            else:
                QMessageBox.warning(self, "Error", "Failed to load past work")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def load_subjects(self):
        try:
            response = self.session.get('http://localhost:5000/api/subjects')
            if response.status_code == 200:
                subjects = response.json()
                self.subject_input.clear()
                if subjects:
                    for subject in subjects:
                        self.subject_input.addItem(subject['name'], subject['id'])
                else:
                    self.subject_input.addItem("No subjects available")
            else:
                QMessageBox.warning(self, "Error", "Failed to load subjects")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def load_topics(self):
        subject_id = self.subject_input.currentData()
        if subject_id is None:
            self.topic_input.clear()
            self.topic_input.addItem("No topics available")
            return

        try:
            response = self.session.get(f'http://localhost:5000/api/topics/{subject_id}')
            if response.status_code == 200:
                topics = response.json()
                self.topic_input.clear()
                if topics:
                    for topic in topics:
                        self.topic_input.addItem(topic['name'], topic['id'])
                else:
                    self.topic_input.addItem("No topics available")
            else:
                QMessageBox.warning(self, "Error", "Failed to load topics")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def upload_and_grade_work(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Upload Homework",
            "",
            "PDF, and DOCX Files (*.pdf *.docx)",
            options=options
        )
        if file_path:
            subject_id = self.subject_input.currentData()
            topic_id = self.topic_input.currentData()

            if not subject_id or not topic_id:
                QMessageBox.warning(self, "Error", "Please select a subject and a topic.")
                return

            try:
                files = {'file': open(file_path, 'rb')}
                data = {'subject_id': subject_id, 'topic_id': topic_id}
                response = self.session.post('http://localhost:5000/api/upload_work', files=files, data=data)
                if response.status_code == 200:
                    result = response.json()
                    self.show_results(result)
                else:
                    error_message = response.json().get("message", "Failed to upload and process homework")
                    QMessageBox.warning(self, "Error", error_message)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred during upload: {e}")

    def show_results(self, result):
        feedback = result.get('feedback', 'No feedback available')
        grade = result.get('grade', 'No grade available')
        self.chat_area.append(f"Feedback: {feedback}")
        self.chat_area.append(f"Grade: {grade}")

    def ask_question(self):
        question = self.query_input.text()
        subject_id = self.subject_input.currentData()
        topic_id = self.topic_input.currentData()

        if question and subject_id and topic_id:
            self.chat_area.append(f"You: {question}")
            self.query_input.clear()

            try:
                response = self.session.post('http://localhost:5000/api/student_query',
                                             json={'subject_id': subject_id, 'topic_id': topic_id,
                                                   'question': question})
                if response.status_code == 200:
                    answer = response.json().get('answer', 'No answer available')
                    self.chat_area.append(f"EduChatBot: {answer}")
                else:
                    QMessageBox.warning(self, "Error", "Failed to get an answer")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred: {e}")
        else:
            QMessageBox.warning(self, "Error", "Please select a subject, topic, and enter a question.")

    def show_help(self):
        QMessageBox.information(self, "Help", "This is the student dashboard. You can upload work, ask questions, and review past submissions.")
