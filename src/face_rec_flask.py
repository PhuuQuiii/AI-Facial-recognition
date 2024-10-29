from flask import Flask, render_template, request, session, jsonify
import numpy as np
import mysql.connector
import align.detect_face
import facenet
import tensorflow as tf
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from dotenv import load_dotenv
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

        connection = mysql.connector.connect(host='localhost', user='root', database='face_recognition')
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM teachers WHERE teacher_id = %s AND password = %s", (teacher_id, password))
        teacher = cursor.fetchone()

        if teacher:
            session['teacher_id'] = teacher_id
            session['teacher_name'] = teacher[3]  # Tên giáo viên
            session['class'] = teacher[4]  # Mã lớp
            return jsonify({"success": True, "name": teacher[3], "class": teacher[4]})
        return jsonify({"success": False, "message": "Mã số giáo viên hoặc mật khẩu không đúng."})
    return render_template('login_teacher.html')  # Trả về trang đăng nhập nếu là GET

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
    # Xử lý thông báo từ face_rec_cam.py
    emit('response', data, broadcast=True)

@app.before_request
def before_request_func():
    print("Received request:", request.url)
    print("Request headers:", request.headers)

if __name__ == '__main__':
    socketio.run(app, debug=True)
