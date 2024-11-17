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
import datetime
import base64
import subprocess
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
def home():
    return render_template('index.html')
@app.route('/student')
def student():
    return render_template('student.html')
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
    if 'MSSV' in data:
        student_info = get_student_info(data['MSSV'])
        if student_info:
            # Lấy class_id từ cơ sở dữ liệu
            class_id = get_class_id_by_student_id(student_info["MSSV"])
            if class_id:
                # Cập nhật điểm danh
                date = str(datetime.date.today())  # Lấy ngày hiện tại
                update_attendance(student_info["MSSV"], class_id, date, "Present")  # Gọi hàm với các tham số cần thiết
                
                emit('response', {
                    "MSSV": student_info["MSSV"],
                    "name": student_info["name"],
                    "message": f"Sinh viên {student_info['name']} (MSSV: {student_info['MSSV']}) đã điểm danh."
                }, broadcast=True)
            else:
                emit('response', {"error": "Class ID not found"}, broadcast=True)
        else:
            emit('response', {"error": "Student not found"}, broadcast=True)
    else:
        emit('response', {"error": "MSSV không được cung cấp"}, broadcast=True)

def get_class_id_by_student_id(student_id): # lấy class_id từ cơ sở dữ liệu
    connection = mysql.connector.connect(host='localhost', user='root', database='face_recognition')
    cursor = connection.cursor()
    cursor.execute("""
        SELECT sc.class_id 
        FROM Student_Class sc 
        WHERE sc.student_id = %s
    """, (student_id,))
    class_id = cursor.fetchone()
    cursor.close()
    connection.close()
    
    if class_id:
        return class_id[0]  # Trả về class_id
    return None  # Nếu không tìm thấy

@app.route('/get_classes')
def get_classes():
    teacher_id = request.args.get('teacher_id')
    
    try:
        connection = mysql.connector.connect(host='localhost', user='root', database='face_recognition')
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    cursor = connection.cursor()
    cursor.execute("SELECT id_class, name, room FROM Class WHERE teacher_id = %s", (teacher_id,))
    classes = cursor.fetchall()
    print("Classes fetched:", classes)
    return jsonify({"classes": [{"id_class": cls[0], "name": cls[1], "room": cls[2]} for cls in classes]})

@app.route('/get_students')
def get_students():
    class_id = request.args.get('class_id')
    connection = mysql.connector.connect(host='localhost', user='root', database='face_recognition')
    cursor = connection.cursor()
    cursor.execute("""
        SELECT s.MSSV, s.name, a.date, a.status 
        FROM Student s
        JOIN Student_Class sc ON s.MSSV = sc.student_id
        JOIN Attendance a ON s.MSSV = a.student_id
        WHERE sc.class_id = %s
    """, (class_id,))
    students = cursor.fetchall()
    return jsonify({"students": [{"MSSV": student[0], "name": student[1], "date": student[2], "status": student[3]} for student in students]})

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

@app.route('/update_attendance', methods=['POST'])
def update_attendance(mssv, class_id, date, status):
    connection = mysql.connector.connect(host='localhost', user='root', database='face_recognition')
    cursor = connection.cursor()
    
    # Xóa bản ghi cũ nếu tồn tại
    cursor.execute("""
        DELETE FROM Attendance 
        WHERE student_id = %s AND date = %s
    """, (mssv, date))

    # Thêm bản ghi mới
    cursor.execute("""
        INSERT INTO Attendance (student_id, class_id, date, status) 
        VALUES (%s, %s, %s, %s) 
        ON DUPLICATE KEY UPDATE status=VALUES(status), date=VALUES(date)
    """, (mssv, class_id, date, status))

    connection.commit()
    cursor.close()
    connection.close()
    
    return jsonify({"success": True})

@app.before_request
def before_request_func():
    print("Received request:", request.url)
    print("Request headers:", request.headers)

@app.route('/check_attendance')
def check_attendance():
    student_id = request.args.get('student_id')
    date = request.args.get('date')

    connection = mysql.connector.connect(host='localhost', user='root', database='face_recognition')
    cursor = connection.cursor()
    cursor.execute("""
        SELECT COUNT(*) 
        FROM Attendance 
        WHERE student_id = %s AND date = %s
    """, (student_id, date))
    count = cursor.fetchone()[0]
    cursor.close()
    connection.close()

    return jsonify({"exists": count > 0})

@app.route('/delete_attendance', methods=['POST'])
def delete_attendance():
    data = request.get_json()
    mssv = data['MSSV']
    date = data['date']

    connection = mysql.connector.connect(host='localhost', user='root', database='face_recognition')
    cursor = connection.cursor()
    cursor.execute("""
        DELETE FROM Attendance 
        WHERE student_id = %s AND date = %s
    """, (mssv, date))
    connection.commit()
    cursor.close()
    connection.close()

    return jsonify({"success": True})

@app.route('/save_images', methods=['POST'])
def save_images():
    data = request.get_json()
    student_id = data['studentId']
    images = data['images']

    # Đường dẫn đến thư mục raw trong thư mục FaceData
    directory = f'E:/DoAN/Facial_recognition/Dataset/FaceData/raw/{student_id}'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Lưu từng ảnh vào thư mục
    for i, image in enumerate(images):
        # Xử lý dữ liệu base64
        image_data = image.split(",")[1]
        try:
            with open(f"{directory}/{student_id}_{i + 1}.jpg", "wb") as fh:
                fh.write(base64.b64decode(image_data))
            print(f"Saved image: {student_id}_{i + 1}.jpg")
        except Exception as e:
            print(f"Error saving image {i + 1}: {e}")

    return jsonify({"success": True})

@app.route('/preprocess_data', methods=['POST'])
def preprocess_data():
    try:
        # Chạy lệnh tiền xử lý
        subprocess.run(['python', 'align_dataset_mtcnn.py', '../Dataset/FaceData/raw', '../Dataset/FaceData/processed', '--image_size', '160', '--margin', '32', '--random_order', '--gpu_memory_fraction', '0.25'], check=True)
        return jsonify({"success": True})
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return jsonify({"success": False, "message": str(e)})

@app.route('/train_model', methods=['POST'])
def train_model():
    try:
        # Chạy lệnh huấn luyện
        subprocess.run(['python', 'classifier.py', 'TRAIN', '../Dataset/FaceData/processed', '../Models/20180402-114759.pb', '../Models/facemodel.pkl', '--batch_size', '1000'], check=True)
        return jsonify({"success": True})
    except Exception as e:
        print(f"Error during training: {e}")
        return jsonify({"success": False, "message": str(e)})

if __name__ == '__main__':
    socketio.run(app, debug=True)
