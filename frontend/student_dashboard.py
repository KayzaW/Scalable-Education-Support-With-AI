from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox
import requests


class StudentDashboard(QWidget):
    def __init__(self, parent=None):
        super(StudentDashboard, self).__init__(parent)
        self.layout = QVBoxLayout()

        self.welcome_label = QLabel("Welcome, Student!")
        self.layout.addWidget(self.welcome_label)

        self.upload_button = QPushButton("Upload Work")
        self.upload_button.clicked.connect(self.upload_work)
        self.layout.addWidget(self.upload_button)

        self.setLayout(self.layout)

    def upload_work(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Upload Work", "", "Images (*.png *.jpg *.jpeg)",
                                                   options=options)
        if file_path:
            files = {'file': open(file_path, 'rb')}
            response = requests.post('http://localhost:5000/grade', files=files)
            if response.status_code == 200:
                result = response.json()
                self.show_results(result)
            else:
                QMessageBox.warning(self, "Error", "Failed to upload and process work")

    def show_results(self, result):
        feedback = result.get('feedback', 'No feedback available')
        grade = result.get('grade', 'No grade available')
        additional_work = result.get('additional_work', 'No additional work available')

        QMessageBox.information(self, "Results",
                                f"Grade: {grade}\nFeedback: {feedback}\nAdditional Work: {additional_work}")
