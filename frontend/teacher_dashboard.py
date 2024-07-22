from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QListWidget, QFileDialog, QMessageBox, QLineEdit, QTreeWidget, QTreeWidgetItem, QComboBox, QInputDialog
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
import requests

class TeacherDashboard(QWidget):
    def __init__(self, parent=None, user_id=None, session=None):
        super(TeacherDashboard, self).__init__(parent)
        self.parent = parent
        self.user_id = user_id
        self.session = session
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        title = QLabel("Teacher Dashboard")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 24))
        main_layout.addWidget(title)


        self.logout_button = QPushButton("Logout")
        self.logout_button.setFont(QFont("Arial", 12))
        self.logout_button.setFixedHeight(40)
        self.logout_button.clicked.connect(self.handle_logout)
        main_layout.addWidget(self.logout_button, alignment=Qt.AlignRight)


        new_user_label = QLabel("Create New User")
        new_user_label.setFont(QFont("Arial", 18))
        main_layout.addWidget(new_user_label)

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Enter username")
        main_layout.addWidget(self.username_input)

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Enter password")
        self.password_input.setEchoMode(QLineEdit.Password)
        main_layout.addWidget(self.password_input)

        self.role_input = QComboBox()
        self.role_input.addItems(["student", "teacher"])
        main_layout.addWidget(self.role_input)

        self.create_user_button = QPushButton("Create User")
        self.create_user_button.clicked.connect(self.create_user)
        main_layout.addWidget(self.create_user_button)


        subject_label = QLabel("Manage Subjects")
        subject_label.setFont(QFont("Arial", 18))
        main_layout.addWidget(subject_label)

        self.subject_input = QComboBox()
        self.load_subjects()
        self.subject_input.currentIndexChanged.connect(self.load_topics)
        main_layout.addWidget(self.subject_input)

        self.new_subject_button = QPushButton("Add New Subject")
        self.new_subject_button.clicked.connect(self.add_new_subject)
        main_layout.addWidget(self.new_subject_button)

        topic_label = QLabel("Manage Topics")
        topic_label.setFont(QFont("Arial", 18))
        main_layout.addWidget(topic_label)

        self.topic_input = QComboBox()
        main_layout.addWidget(self.topic_input)
        self.subject_input.currentIndexChanged.connect(self.load_topics)

        self.new_topic_button = QPushButton("Add New Topic")
        self.new_topic_button.clicked.connect(self.add_new_topic)
        main_layout.addWidget(self.new_topic_button)


        material_label = QLabel("Uploaded Materials")
        material_label.setFont(QFont("Arial", 18))
        main_layout.addWidget(material_label)

        self.material_list = QListWidget()
        main_layout.addWidget(self.material_list)

        self.upload_button = QPushButton("Upload New Material")
        self.upload_button.clicked.connect(self.upload_material)
        main_layout.addWidget(self.upload_button)

        self.delete_button = QPushButton("Delete Selected Material")
        self.delete_button.clicked.connect(self.delete_material)
        main_layout.addWidget(self.delete_button)


        submission_label = QLabel("Student Submissions")
        submission_label.setFont(QFont("Arial", 18))
        main_layout.addWidget(submission_label)

        self.submissions_tree = QTreeWidget()
        self.submissions_tree.setHeaderLabels(["Student", "Subject", "Topic", "Date", "Grade", "Feedback"])
        main_layout.addWidget(self.submissions_tree)

        self.setLayout(main_layout)
        self.load_materials()
        self.load_student_submissions()

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

    def create_user(self):
        username = self.username_input.text()
        password = self.password_input.text()
        role = self.role_input.currentText()

        if not username or not password:
            QMessageBox.warning(self, "Error", "Username and password are required.")
            return

        try:
            response = self.session.post('http://localhost:5000/api/register', json={
                'username': username,
                'password': password,
                'role': role
            })
            if response.status_code == 200:
                QMessageBox.information(self, "Success", "User created successfully.")
                self.username_input.clear()
                self.password_input.clear()
            else:
                QMessageBox.warning(self, "Error", "Failed to create user.")
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

    def add_new_subject(self):
        new_subject, ok = QInputDialog.getText(self, "New Subject", "Enter the name of the new subject:")
        if ok and new_subject:
            try:
                response = self.session.post('http://localhost:5000/api/add_subject', json={'subject': new_subject})
                if response.status_code == 200:
                    QMessageBox.information(self, "Success", "Subject added successfully.")
                    self.load_subjects()
                else:
                    QMessageBox.warning(self, "Error", "Failed to add subject")
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

    def add_new_topic(self):
        subject_id = self.subject_input.currentData()
        new_topic, ok = QInputDialog.getText(self, "New Topic", f"Enter the name of the new topic for {self.subject_input.currentText()}:")
        if ok and new_topic:
            try:
                response = self.session.post('http://localhost:5000/api/add_topic', json={'subject_id': subject_id, 'topic': new_topic})
                if response.status_code == 200:
                    QMessageBox.information(self, "Success", "Topic added successfully.")
                    self.load_topics()
                else:
                    QMessageBox.warning(self, "Error", "Failed to add topic")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def load_materials(self):
        subject_id = self.subject_input.currentData()
        if subject_id is None:
            QMessageBox.information(self, "Information", "No subjects available. Please add a subject first.")
            return

        try:
            response = self.session.get(f'http://localhost:5000/api/materials/{subject_id}')
            if response.status_code == 200:
                materials = response.json()
                self.material_list.clear()
                if materials:
                    for material in materials:
                        self.material_list.addItem(f"{material['file_path']} - {material['created_at']}")
                else:
                    self.material_list.addItem("No materials uploaded for this subject.")
            else:
                QMessageBox.warning(self, "Error", "Failed to load materials")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def load_student_submissions(self):
        try:
            response = self.session.get('http://localhost:5000/api/all_submissions')
            if response.status_code == 200:
                submissions_by_student = response.json()
                self.submissions_tree.clear()
                if submissions_by_student:
                    for student, submissions in submissions_by_student.items():
                        student_item = QTreeWidgetItem([student])
                        self.submissions_tree.addTopLevelItem(student_item)
                        for submission in submissions:
                            submission_item = QTreeWidgetItem([
                                "",
                                submission['subject'],
                                submission['topic'],
                                submission['created_at'],
                                submission['grade'],
                                submission['feedback']
                            ])
                            student_item.addChild(submission_item)
                else:
                    QMessageBox.information(self, "Information", "No submissions available.")
            else:
                QMessageBox.warning(self, "Error", "Failed to load student submissions")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def upload_material(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Material", "", "All Files (*);;PDF Files (*.pdf);;Word Files (*.doc *.docx)", options=options
        )
        if file_path:
            try:
                subject_id = self.subject_input.currentData()
                topic_id = self.topic_input.currentData()
                content_type_options = ["text", "question bank", "mark scheme"]

                content_type, ok = QInputDialog.getItem(self, "Select Content Type",
                                                        "Content Type:", content_type_options, 0, False)
                if not ok or not content_type:
                    QMessageBox.warning(self, "Error", "Content type is required.")
                    return

                with open(file_path, 'rb') as file:
                    files = {'file': file}
                    data = {'subject_id': subject_id, 'topic_id': topic_id, 'content_type': content_type}
                    response = self.session.post('http://localhost:5000/api/upload_material', files=files, data=data)

                if response.status_code == 201:
                    QMessageBox.information(self, "Success", "Material uploaded successfully")
                    self.load_materials()
                else:
                    QMessageBox.warning(self, "Error",
                                        f"Failed to upload material: {response.json().get('message', 'Unknown error')}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred during file upload: {e}")

    def delete_material(self):
        selected_item = self.material_list.currentItem()
        if selected_item:
            filename = selected_item.text().split(' - ')[0]
            try:
                response = self.session.get(f'http://localhost:5000/api/materials/{self.subject_input.currentData()}')
                materials = response.json()
                material_id = None
                for material in materials:
                    if material['file_path'] == filename:
                        material_id = material['id']
                        break
                if material_id:
                    delete_response = self.session.delete(f'http://localhost:5000/api/delete_material/{material_id}')
                    if delete_response.status_code == 200:
                        QMessageBox.information(self, "Success", "Material deleted successfully")
                        self.load_materials()
                    else:
                        QMessageBox.warning(self, "Error", "Failed to delete material")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred: {e}")
        else:
            QMessageBox.warning(self, "Error", "No material selected")
