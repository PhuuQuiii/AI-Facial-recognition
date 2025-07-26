from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from imutils.video import VideoStream
import argparse
import facenet
import imutils
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
print("Current working directory:", os.getcwd())
print("face_rec_cam.py location:", os.path.abspath(__file__))
cdc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../CDCNpp'))
print("CDC path to add:", cdc_path)
sys.path.append(cdc_path)
print("sys.path:", sys.path)
from CDCNpp.main import load_model as load_cdc_model, predict_on_worker as cdc_predict
import math
import pickle
import align.detect_face
import numpy as np
import cv2
import collections
from sklearn.svm import SVC
from flask import Flask
from flask_socketio import SocketIO
from socketio import Client
from dotenv import load_dotenv
import datetime
import socketio

# Khởi tạo Flask app
app = Flask(__name__)
# app.secret_key = os.getenv('SECRET_KEY')
app.secret_key = 'vaa_nckh_2025_secret'
socketio_client = socketio.Client()

attendance_success = False



# attendance_log = {}

# def check_attendance(MSSV):
#     today = str(datetime.date.today())

#     # Nếu ngày hôm nay chưa tồn tại trong attendance_log, khởi tạo một set mới
#     if today not in attendance_log:
#         attendance_log[today] = set()

#     # Kiểm tra nếu MSSV đã có trong set của ngày hôm nay
#     if MSSV in attendance_log[today]:
#         print(f"Sinh viên {MSSV} đã điểm danh hôm nay.")
#         return False
#     else:
#         # Nếu chưa điểm danh, thêm MSSV vào set và trả về True
#         attendance_log[today].add(MSSV)
#         print(f"Sinh viên {MSSV} được điểm danh.")
#         return True


@socketio_client.on('stop_camera')
def on_stop_camera(data):
    print("Stopping camera...")
    sys.exit(0)

def main():

    attendance_success = False

    # Khởi tạo model CDCNpp
    # cdc_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../CDCNpp/CDCNpp_BinaryMask_P1_07_60.pkl'))
    cdc_model_path = r'D:\NHATTRUONG\VAA\NCKH\AI-Facial-recognition\CDCNpp\CDCNpp_BinaryMask_P1_07_30.pkl'
    cdc_model, cdc_device = load_cdc_model(cdc_model_path)
    print("CDCNpp model loaded successfully.")

    # Kết nối đến server
    try:
        socketio_client.connect('http://127.0.0.1:5000')
        print("Connected to server successfully.")
    except Exception as e:
        print(f"Connection error: {e}")
        return

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path of the video you want to test on.', default=0)
    # parser.add_argument("date", type=str, help="Date in YYYY-MM-DD format")
    # parser.add_argument("classId", type=str, help="Class ID")
    args = parser.parse_args()
    
    # date = args.date
    # classId = args.classId

    MINSIZE = 20 # nếu khuôn mặt nhỏ hơn kích thước 20 pixel ( bỏ qua )
    THRESHOLD = [0.6, 0.7, 0.7] # quét qua hình ảnh và đưa ra các dự đoán về các vùng có thể chứa khuôn mặt.
    FACTOR = 0.709 # được sử dụng để giảm kích thước hình ảnh trong quá trình quét ( < MINSIZE = 20 thì ngừng quét)
    # IMAGE_SIZE = 182
    INPUT_IMAGE_SIZE = 160 # kích thước mà các khuôn mặt sẽ được thay đổi kích thước trước khi được đưa vào mô hình để trích xuất đặc trưng.
    # CLASSIFIER_PATH = '../Models/facemodel.pkl'
    # CLASSIFIER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Models/facemodel.pkl'))
    CLASSIFIER_PATH = 'Models/facemodel.pkl'
    VIDEO_PATH = args.path
    # FACENET_MODEL_PATH = '../Models/20180402-114759.pb' # Tệp Protobuf là tệp tuần tự hóa dữ liệu, tối ưu cho việc lưu trữ và trao đổi dữ liệu nhanh chóng.
    # FACENET_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Models/20180402-114759.pb'))
    FACENET_MODEL_PATH = 'Models/20180402-114759.pb'

    # Tải mô hình phân loại khuôn mặt từ file pickle
    with open(CLASSIFIER_PATH, 'rb') as file:
        model, class_names = pickle.load(file)
    print("Custom Classifier, Successfully loaded")

    with tf.Graph().as_default():
        #  Khởi tạo một phiên TensorFlow với cấu hình GPU.
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default(): # khởi tạo một phiên làm việc (session) cho mô hình Facenet và lấy các tensor cần thiết:
            #  Tải mô hình Facenet để trích xuất đặc trưng khuôn mặt.
            print('Loading feature extraction model')
            facenet.load_model(FACENET_MODEL_PATH)

            # Lấy các tensor đầu vào và đầu ra từ mô hình.
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0") # Đây là tensor đầu vào của mô hình, được sử dụng để nhận các hình ảnh khuôn mặt mà bạn muốn xử lý. Tensor này thường có kích thước (None, 160, 160, 3), trong đó 160 là kích thước của hình ảnh (chiều cao và chiều rộng), và 3 là số kênh màu (RGB).
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0") # Đây là tensor đầu ra của mô hình, chứa các đặc trưng (embeddings) của khuôn mặt sau khi được xử lý. Mỗi khuôn mặt sẽ được biểu diễn bằng một vector đặc trưng có kích thước cố định (ví dụ: 128 hoặc 512 chiều, tùy thuộc vào cấu hình của mô hình).
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0") # Đây là tensor được sử dụng để chỉ định chế độ hoạt động của mô hình (huấn luyện hoặc suy diễn). Khi phase_train_placeholder được đặt là False, mô hình sẽ hoạt động ở chế độ suy diễn, không cập nhật trọng số.
            embedding_size = embeddings.get_shape()[1] # Đây là kích thước của vector đặc trưng (embeddings) mà mô hình tạo ra. Nó được lấy từ hình dạng của tensor embeddings, cho phép bạn biết số chiều của vector đặc trưng mà bạn sẽ làm việc với.

            #  Tạo các mạng MTCNN để phát hiện khuôn mặt.
            align_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'align'))
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, align_dir)

            # Khởi tạo các biến để theo dõi người đã phát hiện và bắt đầu video stream từ camera.
            people_detected = set()
            person_detected = collections.Counter()
            cap = VideoStream(src=0).start()

            while True:
                # Vòng lặp chính để đọc từng khung hình từ camera, thay đổi kích thước và lật khung hình.
                frame = cap.read()
                frame = imutils.resize(frame, width=600)
                frame = cv2.flip(frame, 1) # 1: Lật ngang (trái <-> phải)-- lật ngang sẽ giúp hiển thị đúng như nhìn qua gương

                # Phát hiện khuôn mặt trong khung hình.
                bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)


                # Kiểm tra số lượng khuôn mặt phát hiện được và hiển thị thông báo nếu có nhiều hơn một khuôn mặt.
                faces_found = bounding_boxes.shape[0]
                print("Số khuôn mặt phát hiện:", faces_found)
                try:
                    if faces_found > 1:
                        cv2.putText(frame, "Only one face", (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (255, 255, 255), thickness=1, lineType=2)
                    
                    # Nếu có khuôn mặt được phát hiện, lưu trữ tọa độ của chúng.
                    elif faces_found > 0:
                        det = bounding_boxes[:, 0:4]
                        bb = np.zeros((faces_found, 4), dtype=np.int32)
                        for i in range(faces_found):
                            bb[i][0] = det[i][0]
                            bb[i][1] = det[i][1]
                            bb[i][2] = det[i][2]
                            bb[i][3] = det[i][3]
                            # print(bb[i][3]-bb[i][1])
                            # print(frame.shape[0])
                            # print((bb[i][3]-bb[i][1])/frame.shape[0])

                            # Cắt và tiền xử lý khuôn mặt nếu chiều cao của khuôn mặt lớn hơn 25% chiều cao khung hình.
                            if (bb[i][3]-bb[i][1])/frame.shape[0]>0.1:
                                cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]

                                # --- Kiểm tra thật/giả bằng CDCNpp ---
                                result = cdc_predict(cdc_model, cropped, cdc_device)
                                if isinstance(result, tuple) and len(result) == 2:
                                    real_or_fake, prob = result
                                else:
                                    print(f"CDC Predict Error: {result}")
                                    real_or_fake, prob = "Error", 0.0
                                if real_or_fake != "Real":
                                    name = f"Fake ({prob:.2f})"
                                    color = (0, 0, 255)
                                else:
                                    name = f"Real ({prob:.2f})"
                                    color = (0, 255, 0)

                                # Vẽ khung và chữ trên viền khuôn mặt
                                cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), color, 2)
                                cv2.putText(frame, name, (bb[i][0], bb[i][1] - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2, lineType=2)

                                if real_or_fake != "Real":
                                    continue  # Bỏ qua nhận diện nếu là mặt giả
                                
                                # Nếu là mặt thật, tiếp tục nhận diện như cũ
                                scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                                 interpolation=cv2.INTER_CUBIC)
                                scaled = facenet.prewhiten(scaled)
                                scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)

                                # Trích xuất đặc trưng khuôn mặt bằng cách chạy mô hình
                                feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False} # Đặt False ( ở chế độ phân loại)
                                emb_array = sess.run(embeddings, feed_dict=feed_dict) # Phương thức này sẽ chạy tensor embeddings trong phiên làm việc TensorFlow, trả về các đặc trưng khuôn mặt.

                                # Dự đoán khuôn mặt bằng mô hình SVM và lấy tên của người được nhận diện.
                                predictions = model.predict_proba(emb_array)
                                best_class_indices = np.argmax(predictions, axis=1)
                                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                                best_class_probabilities = predictions[
                                    np.arange(len(best_class_indices)), best_class_indices]
                                best_name = class_names[best_class_indices[0]]
                                print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))

                                # Nếu xác suất dự đoán lớn hơn 0.8, vẽ hình chữ nhật quanh khuôn mặt và hiển thị tên cùng xác suất. Gửi thông tin điểm danh qua SocketIO.
                                if best_class_probabilities > 0.8:
                                    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                                    text_x = bb[i][0]
                                    text_y = bb[i][3] + 20
                                    name = class_names[best_class_indices[0]]
                                    cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (255, 255, 255), thickness=1, lineType=2)
                                    cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y + 17),
                                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), thickness=1, lineType=2)
                                    
                                    socketio_client.emit('response', {"success": True,"MSSV": best_name, "date": date, "classId": classId},)
                                    attendance_success = True

                                    # # Gửi thông báo qua SocketIO nếu chưa điểm danh hôm nay
                                    # if check_attendance(best_name):
                                    #     socketio_client.emit('response', {"MSSV": best_name})
                                    #     print(f"Đã gửi socket điểm danh cho sinh viên: {name} (MSSV: {best_name})")
                                    # else:
                                    #     print(f"Sinh viên {name} (MSSV: {best_name}) đã điểm danh trước đó.")

                                else:
                                    name = "Unknown"
                except Exception as e:
                    print(f"Error processing frame: {e}")

                cv2.imshow('Face Recognition', frame)
                # if cv2.waitKey(1) & 0xFF == ord('q') or attendance_success:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
