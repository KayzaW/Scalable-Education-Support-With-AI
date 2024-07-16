from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton

class TeacherPage(QWidget):
    def __init__(self, parent=None):
        super(TeacherPage, self).__init__(parent)
        self.layout = QVBoxLayout()

        self.welcome_label = QLabel("Welcome, Teacher!")
        self.layout.addWidget(self.welcome_label)

        self.view_students_button = QPushButton("View Students")
        self.layout.addWidget(self.view_students_button)

        self.view_performance_button = QPushButton("View Class Performance")
        self.layout.addWidget(self.view_performance_button)

        self.setLayout(self.layout)
