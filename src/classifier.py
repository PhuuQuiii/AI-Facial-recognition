# Sử dụng tập dữ liệu của riêng bạn để huấn luyện một bộ phân loại (classifier) nhận dạng khuôn mặt

# 2. `main(args)`:
#    - Đây là hàm chính xử lý logic của chương trình, với các đối số (`args`) được truyền vào từ dòng lệnh.

# 3. Thiết lập đồ thị và phiên làm việc: 
#    - `with tf.Graph().as_default()` tạo một đồ thị mới cho TensorFlow.
#    - `with tf.compat.v1.Session() as sess` tạo một phiên làm việc (session) để chạy các phép toán TensorFlow.

# 4. Xử lý tập dữ liệu:
#    - `facenet.get_dataset(args.data_dir)` tải dữ liệu từ thư mục `data_dir`. 
#    - Nếu `args.use_split_dataset` là `True`, tập dữ liệu được chia thành `train_set` (tập huấn luyện) và `test_set` (tập kiểm tra) bằng hàm `split_dataset`.

# 5. Xác thực số lượng ảnh:
#    - Kiểm tra để đảm bảo mỗi lớp có ít nhất một ảnh trong tập dữ liệu với `assert(len(cls.image_paths)>0)`.

# 6. Lấy đường dẫn ảnh và nhãn:
#    - `facenet.get_image_paths_and_labels(dataset)` trả về hai mảng chứa đường dẫn đến ảnh và nhãn lớp tương ứng.

# 7. Tải mô hình nhận dạng khuôn mặt:
#    - `facenet.load_model(args.model)` tải mô hình đặc trưng đã huấn luyện sẵn, giúp chuyển đổi ảnh thành biểu diễn đặc trưng (embedding).

# 8. Thiết lập Tensor và Placeholders:
#    - `images_placeholder`, `embeddings`, và `phase_train_placeholder` là các Tensor và placeholders trong mô hình đã tải, phục vụ cho việc xử lý ảnh và tính toán đặc trưng.

# 9. Tính toán Embedding:
#    - Chạy ảnh qua mô hình để tạo ra embeddings - các biểu diễn đặc trưng cho từng ảnh. Các biểu diễn này được lưu vào `emb_array`.

# 10. Huấn luyện hoặc phân loại:
#     - Nếu `args.mode` là `'TRAIN'`, mã sẽ huấn luyện một bộ phân loại (classifier) SVM sử dụng các embeddings và nhãn của ảnh. 
#     - Bộ phân loại và tên các lớp được lưu vào một file `.pkl` để sử dụng sau.
#     - Nếu `args.mode` là `'CLASSIFY'`, mã sẽ tải bộ phân loại từ file `.pkl`, dự đoán lớp của các ảnh, và tính toán độ chính xác.

#  Các hàm hỗ trợ

# - `split_dataset()`:
#   - Hàm này chia tập dữ liệu thành hai phần: tập huấn luyện và tập kiểm tra dựa trên số lượng ảnh tối thiểu `min_nrof_images_per_class` và số ảnh huấn luyện `nrof_train_images_per_class` cho mỗi lớp.

# - `parse_arguments(argv)`:
#   - Hàm này xử lý các đối số dòng lệnh, bao gồm chế độ (`mode`), thư mục dữ liệu (`data_dir`), mô hình (`model`), và các thông số khác như kích thước ảnh (`image_size`) và kích thước lô (`batch_size`).

#  Giải thích chi tiết các lệnh

# - `tf.compat.v1.get_default_graph().get_tensor_by_name(...)`: lấy các tensor đã được định nghĩa trong mô hình đã tải.
# - `sess.run(...)`: chạy tính toán trên đồ thị TensorFlow, giúp tính toán embeddings cho ảnh đầu vào.
# - `pickle.dump(...)` và `pickle.load(...)`: dùng để lưu và tải mô hình phân loại dưới dạng file `.pkl`.

#  Đoạn mã này cung cấp quy trình cho việc huấn luyện và thử nghiệm bộ phân loại nhận dạng khuôn mặt bằng cách sử dụng mô hình nhận dạng đặc trưng khuôn mặt (FaceNet).

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC

def main(args):
  
    with tf.Graph().as_default():
      
        with tf.compat.v1.Session() as sess:
            
            np.random.seed(seed=args.seed)
            
            if args.use_split_dataset:
                dataset_tmp = facenet.get_dataset(args.data_dir)
                train_set, test_set = split_dataset(dataset_tmp, args.min_nrof_images_per_class, args.nrof_train_images_per_class)
                if (args.mode=='TRAIN'):
                    dataset = train_set
                elif (args.mode=='CLASSIFY'):
                    dataset = test_set
            else:
                dataset = facenet.get_dataset(args.data_dir)

            # Check that there are at least one training image per class
            for cls in dataset:
                assert(len(cls.image_paths)>0, 'There must be at least one image for each class in the dataset')

                 
            paths, labels = facenet.get_image_paths_and_labels(dataset)
            
            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))
            
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(args.model)
            
            # Get input and output tensors
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            print(embedding_size)
            
            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i*args.batch_size
                end_index = min((i+1)*args.batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, args.image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
            
            classifier_filename_exp = os.path.expanduser(args.classifier_filename)

            if (args.mode=='TRAIN'): # Sử dụng SVM
                # Train classifier
                print('Training classifier')
                model = SVC(kernel='linear', probability=True)
                model.fit(emb_array, labels)
            
                # Create a list of class names
                class_names = [ cls.name.replace('_', ' ') for cls in dataset]

                # Saving classifier model
                with open(classifier_filename_exp, 'wb') as outfile:
                    pickle.dump((model, class_names), outfile)
                print('Saved classifier model to file "%s"' % classifier_filename_exp)
                # Sau khi huấn luyện xong, mô hình SVM cùng với tên các lớp (class_names) sẽ được lưu vào file bằng pickle để sử dụng sau này.
                
            elif (args.mode=='CLASSIFY'): # Tải mô hình SVM đã lưu từ file classifier_filename_exp bằng pickle.
                # Classify images
                print('Testing classifier')
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)

                print('Loaded classifier model from file "%s"' % classifier_filename_exp)

                predictions = model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1) # Lấy chỉ số của lớp dự đoán tốt nhất với np.argmax(predictions, axis=1) để xác định lớp có xác suất cao nhất.
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                
                for i in range(len(best_class_indices)): # In ra tên lớp và xác suất dự đoán của lớp đó.
                    print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
                    
                accuracy = np.mean(np.equal(best_class_indices, labels)) # Tính độ chính xác so với nhãn thực (labels) 
                print('Accuracy: %.3f' % accuracy)
                
            
def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        # Remove classes with less than min_nrof_images_per_class
        if len(paths)>=min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
    return train_set, test_set

            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('mode', type=str, choices=['TRAIN', 'CLASSIFY'],
        help='Indicates if a new classifier should be trained or a classification ' + 
        'model should be used for classification', default='CLASSIFY')
    parser.add_argument('data_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('classifier_filename', 
        help='Classifier model file name as a pickle (.pkl) file. ' + 
        'For training this is the output and for classification this is an input.')
    parser.add_argument('--use_split_dataset', 
        help='Indicates that the dataset specified by data_dir should be split into a training and test set. ' +  
        'Otherwise a separate test set can be specified using the test_data_dir option.', action='store_true')
    parser.add_argument('--test_data_dir', type=str,
        help='Path to the test data directory containing aligned images used for testing.')
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--min_nrof_images_per_class', type=int,
        help='Only include classes with at least this number of images in the dataset', default=20)
    parser.add_argument('--nrof_train_images_per_class', type=int,
        help='Use this number of images from each class for training and the rest for testing', default=10)
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
