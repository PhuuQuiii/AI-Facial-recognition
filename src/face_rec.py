# Đoạn mã bạn cung cấp là một ứng dụng nhận diện khuôn mặt sử dụng TensorFlow và OpenCV. Dưới đây là giải thích chi tiết cho từng phần của mã:

#  2. Nhập các thư viện cần thiết

# - Các thư viện được nhập vào để phục vụ cho các chức năng như:
#   - `argparse`: Để xử lý các tham số dòng lệnh.
#   - `facenet`: Mô hình nhận diện khuôn mặt.
#   - `cv2`: Thư viện OpenCV để xử lý hình ảnh và video.
#   - `mysql.connector`: Để kết nối và tương tác với cơ sở dữ liệu MySQL.
#   - `json`: Để xử lý dữ liệu JSON.

#  3. Hàm chính

# - Tạo một đối tượng `ArgumentParser` để nhận đường dẫn video từ người dùng. Nếu không có, mặc định là sử dụng webcam (0).

#  4. Thiết lập các tham số cần thiết

# - Thiết lập các tham số cho việc phát hiện và nhận diện khuôn mặt như kích thước tối thiểu, ngưỡng phát hiện, kích thước đầu vào của hình ảnh, đường dẫn tới mô hình classifier và mô hình Facenet.

#  5. Tải mô hình đã huấn luyện

# - Tải mô hình classifier đã huấn luyện từ tệp pickle.

#  6. Khởi tạo TensorFlow

# - Tạo một đồ thị TensorFlow mới và khởi tạo phiên làm việc. Tùy chọn GPU để giới hạn bộ nhớ GPU sử dụng.

#  7. Tải mô hình MTCNN

# - Tải mô hình MTCNN (Multi-task Cascaded Convolutional Networks) để phát hiện khuôn mặt.

#  8. Lấy tensor đầu vào và đầu ra

# - Lấy các tensor cần thiết cho đầu vào (hình ảnh) và đầu ra (embedding khuôn mặt).

#  9. Khởi tạo MTCNN

# - Tạo các mạng MTCNN (P-Net, R-Net, O-Net) để phát hiện khuôn mặt.

#  10. Đọc video và xử lý từng khung hình

# - Mở video từ đường dẫn đã chỉ định và đọc từng khung hình. Dùng MTCNN để phát hiện khuôn mặt trong từng khung hình.

#  11. Xử lý các khuôn mặt được phát hiện

# - Nếu phát hiện được khuôn mặt, lấy bounding box và cắt khuôn mặt ra. Sau đó, thực hiện nhận diện bằng cách tính toán embedding và dự đoán lớp tương ứng.

#  12. Hiển thị kết quả trên khung hình

# - Nếu khuôn mặt được nhận diện với độ chính xác cao, hiển thị tên trên khung hình và vẽ khung bao quanh khuôn mặt.

#  13. Lưu embedding vào cơ sở dữ liệu

# - Nếu độ chính xác cao hơn 0.8, lưu embedding của khuôn mặt vào cơ sở dữ liệu thông qua hàm `save_embedding`.

#  14. Hàm lưu embedding vào cơ sở dữ liệu

# - Kết nối đến cơ sở dữ liệu MySQL, kiểm tra xem tên đã tồn tại chưa. Nếu chưa, lưu embedding vào bảng `students`.

#  15. Kết thúc chương trình

# - Giải phóng tài nguyên khi kết thúc, đóng webcam hoặc video.

# Đoạn mã này tạo ra một hệ thống nhận diện khuôn mặt tự động, có khả năng phát hiện và nhận diện khuôn mặt từ video. Hệ thống sử dụng mô hình MTCNN để phát hiện khuôn mặt và mô hình Facenet để tạo ra các embedding khuôn mặt, từ đó so sánh và lưu trữ thông tin vào cơ sở dữ liệu MySQL.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import argparse
import facenet
import os
import sys
import math
import pickle
import align.detect_face
import numpy as np
import cv2
import collections
from sklearn.svm import SVC
import mysql.connector
import json
from flask import Flask, render_template
from tensorflow.io import gfile



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path of the video you want to test on.', default=0)
    args = parser.parse_args()
    
    # Cai dat cac tham so can thiet
    MINSIZE = 20
    THRESHOLD = [0.6, 0.7, 0.7]
    FACTOR = 0.709
    IMAGE_SIZE = 182
    INPUT_IMAGE_SIZE = 160
    CLASSIFIER_PATH = 'Models/facemodel.pkl'
    VIDEO_PATH = args.path
    FACENET_MODEL_PATH = 'Models/20180402-114759.pb'

    # Load model da train de nhan dien khuon mat - thuc chat la classifier
    with open(CLASSIFIER_PATH, 'rb') as file:
        model, class_names = pickle.load(file)
    print("Custom Classifier, Successfully loaded")

    with tf.Graph().as_default():

        # Cai dat GPU neu co
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():

            # Load model MTCNN phat hien khuon mat
            print('Loading feature extraction model')
            facenet.load_model(FACENET_MODEL_PATH)

            # Lay tensor input va output
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Cai dat cac mang con
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "src/align")

            people_detected = set()
            person_detected = collections.Counter()

            # Lay hinh anh tu file video
            cap = cv2.VideoCapture(VIDEO_PATH)

            while (cap.isOpened()):
                # Doc tung frame
                ret, frame = cap.read()

                # Phat hien khuon mat, tra ve vi tri trong bounding_boxes
                bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

                faces_found = bounding_boxes.shape[0]
                try:
                    if faces_found > 0:
                        det = bounding_boxes[:, 0:4]
                        bb = np.zeros((faces_found, 4), dtype=np.int32)
                        for i in range(faces_found):
                            bb[i][0] = det[i][0]
                            bb[i][1] = det[i][1]
                            bb[i][2] = det[i][2]
                            bb[i][3] = det[i][3]

                            # Cat phan khuon mat tim duoc
                            cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :] # Cắt một phần của hình ảnh (chứa khuôn mặt) từ khung hình (frame)
                            scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), # Thay đổi kích thước của hình ảnh khuôn mặt để phù hợp với kích thước đầu vào mà mô hình yêu cầu.
                                                interpolation=cv2.INTER_CUBIC)
                            scaled = facenet.prewhiten(scaled) # Chuẩn hóa hình ảnh trước khi đưa vào mô hình để cải thiện hiệu suất
                            scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3) #Thay đổi hình dạng của ảnh để phù hợp với đầu vào của mô hình học sâu
                            feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False} # Chuẩn bị dữ liệu đầu vào cho mô hình
                            emb_array = sess.run(embeddings, feed_dict=feed_dict) # Chạy mô hình để tính toán embedding của khuôn mặt
                            
                            # Dua vao model SVM để phân loại
                            predictions = model.predict_proba(emb_array)
                            best_class_indices = np.argmax(predictions, axis=1)
                            best_class_probabilities = predictions[
                                np.arange(len(best_class_indices)), best_class_indices]
                            
                            # Lay ra ten va ty le % cua class co ty le cao nhat
                            best_name = class_names[best_class_indices[0]]
                            print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))

                            # Ve khung mau xanh quanh khuon mat
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                            text_x = bb[i][0]
                            text_y = bb[i][3] + 20

                            # Neu ty le nhan dang > 0.5 thi hien thi ten
                            if best_class_probabilities > 0.8:
                                name = best_name
                                print(f"Recognized: {name} with probability: {best_class_probabilities[0]}")
                                
                                
                            else:
                                name = "Unknown"
                                print("Unknown face detected, not saving embedding.")
                                
                            # Viet text len tren frame    
                            cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (255, 255, 255), thickness=1, lineType=2)
                            cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y + 17),
                                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (255, 255, 255), thickness=1, lineType=2)
                            person_detected[best_name] += 1
                    else:
                        print("No faces found in the frame.")

                except Exception as e:
                    print(f"Error processing frame: {e}")

                # Hien thi frame len man hinh
                cv2.imshow('Face Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Thay đổi cách dừng webcam
            cap.release()  # Nếu bạn sử dụng cv2.VideoCapture
            # cap.stop()  # Nếu bạn sử dụng imutils.VideoStream
            cv2.destroyAllWindows()
            
main()
