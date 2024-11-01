from flask import Flask, render_template, request, session, jsonify
import numpy as np
import mysql.connector
import align.detect_face
import facenet
import tensorflow as tf
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from dotenv import load_dotenv
load_dotenv() 
import os
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY') # Thay đổi secret key cho bảo mật
CORS(app)  # Để cho phép tất cả các nguồn kết nối
socketio = SocketIO(app, cors_allowed_origins="*")

# Các biến toàn cục cần thiết
MINSIZE = 20
THRESHOLD = [0.6, 0.7, 0.7]
FACTOR = 0.709
INPUT_IMAGE_SIZE = 160

# Khởi tạo TensorFlow và model FaceNet
tf.compat.v1.disable_eager_execution()
sess = tf.compat.v1.Session()
facenet.load_model('../Models/20180402-114759.pb')

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/login_teacher', methods=['GET', 'POST'])
def login_teacher():
    if request.method == 'POST':
        data = request.get_json()
        teacher_id = data['teacherId']
        password = data['password']

        # Kết nối đến cơ sở dữ liệu
        connection = mysql.connector.connect(host='localhost', user='root', database='face_recognition')
        cursor = connection.cursor()

        # Truy vấn thông tin giáo viên
        cursor.execute("SELECT * FROM Teacher WHERE MSGV = %s", (teacher_id,))
        teacher = cursor.fetchone()

        if teacher:
            # Kiểm tra mật khẩu
            if teacher[2] == password:  # So sánh với trường password
                session['teacher_id'] = teacher[0]  # MSGV
                session['teacher_name'] = teacher[1]  # Tên giáo viên

                # Lấy thông tin lớp mà giáo viên phụ trách
                cursor.execute("SELECT id_class, name FROM Class WHERE teacher_id = %s", (teacher_id,))
                class_info = cursor.fetchone()

                if class_info:
                    session['class'] = class_info[0]  # Mã lớp
                    session['class_name'] = class_info[1]  # Tên lớp
                    return jsonify({"success": True, "name": teacher[1], "class": class_info[1]})  # Trả về tên lớp

                return jsonify({"success": True, "name": teacher[1], "class": None})

        return jsonify({"success": False, "message": "Mã số giáo viên hoặc mật khẩu không đúng."})

    return render_template('login_teacher.html')



# Route cho trang dashboard giáo viên
@app.route('/teacher_dashboard')
def teacher_dashboard():
    return render_template('teacher_dashboard.html')

@socketio.on('connect')
def handle_connect():
    print("Client đã kết nối")

@socketio.on('disconnect')
def handle_disconnect():
    print("Client đã ngắt kết nối")

@socketio.on('response')
def handle_response(data):
    # Kiểm tra xem dữ liệu có chứa MSSV không
    if 'MSSV' in data:
        student_info = get_student_info(data['MSSV'])
        emit('response', student_info, broadcast=True)
    else:
        emit('response', {"error": "MSSV không được cung cấp"}, broadcast=True)

def get_student_info(student_id):
    connection = mysql.connector.connect(host='localhost', user='root', database='face_recognition')
    cursor = connection.cursor()
    cursor.execute("SELECT MSSV, name FROM Student WHERE MSSV = %s", (student_id,))
    student = cursor.fetchone()
    if student:
        return {
            "MSSV": student[0],
            "name": student[1],
        }
    return {"error": "Student not found"}


@app.before_request
def before_request_func():
    print("Received request:", request.url)
    print("Request headers:", request.headers)

if __name__ == '__main__':
    socketio.run(app, debug=True)
