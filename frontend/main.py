from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget
from login_window import LoginWindow
from student_dashboard import StudentDashboard
from teacher_dashboard import TeacherDashboard

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('EduChatBot')
        self.setGeometry(100, 100, 800, 600)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.login_window = LoginWindow(self)
        self.student_dashboard = None
        self.teacher_dashboard = None

        self.stack.addWidget(self.login_window)

    def show_student_dashboard(self, user_id, session):
        self.student_dashboard = StudentDashboard(self, user_id, session)
        self.stack.addWidget(self.student_dashboard)
        self.stack.setCurrentWidget(self.student_dashboard)

    def show_teacher_dashboard(self, user_id, session):
        self.teacher_dashboard = TeacherDashboard(self, user_id, session)
        self.stack.addWidget(self.teacher_dashboard)
        self.stack.setCurrentWidget(self.teacher_dashboard)

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
