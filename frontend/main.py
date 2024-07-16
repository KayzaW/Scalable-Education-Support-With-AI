import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget
from frontend.login import LoginWindow
from frontend.student_dashboard import StudentDashboard
from frontend.teacher_dashboard import TeacherDashboard

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Educational Chatbot App")
        self.setGeometry(100, 100, 800, 600)

        self.stack = QStackedWidget(self)
        self.setCentralWidget(self.stack)

        self.login_window = LoginWindow(self)
        self.student_dashboard = StudentDashboard(self)
        self.teacher_dashboard = TeacherDashboard(self)

        self.stack.addWidget(self.login_window)
        self.stack.addWidget(self.student_dashboard)
        self.stack.addWidget(self.teacher_dashboard)

        self.show_login()

    def show_login(self):
        self.stack.setCurrentWidget(self.login_window)

    def show_student_dashboard(self):
        self.stack.setCurrentWidget(self.student_dashboard)

    def show_teacher_dashboard(self):
        self.stack.setCurrentWidget(self.teacher_dashboard)

def main():
    app = QApplication(sys.argv)
    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
