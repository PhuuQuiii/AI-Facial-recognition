# 1. Xử lý dữ liệu khuôn mặt:
#    - Đọc tập dữ liệu hình ảnh khuôn mặt từ thư mục được chỉ định.
#    - Sử dụng một mô hình đã được huấn luyện trước (pretrained model) để tạo ra các embedding cho mỗi hình ảnh khuôn mặt.

# 2. Tính toán các chỉ số lọc:
#    - Tính toán trung tâm (center) cho mỗi lớp (mỗi người) trong không gian embedding.
#    - Tính khoảng cách từ mỗi embedding đến trung tâm của lớp tương ứng.

# 3. Đánh giá chất lượng dữ liệu:
#    - Sử dụng các chỉ số đã tính toán để đánh giá chất lượng của các hình ảnh trong dataset.
#    - Điều này có thể giúp xác định các hình ảnh không phù hợp hoặc có chất lượng kém.

# 4. Lưu trữ kết quả:
#    - Ghi các thông tin đã tính toán vào một file HDF5, bao gồm tên lớp, danh sách hình ảnh, danh sách nhãn và khoảng cách đến trung tâm.

# 5. Tiền xử lý dữ liệu:
#    - Script này có thể được sử dụng như một công cụ tiền xử lý để cải thiện chất lượng của tập dữ liệu khuôn mặt trước khi sử dụng cho các tác vụ nhận dạng khuôn mặt.

# Trong không gian embedding, trung tâm là vector trung bình của tất cả các embedding thuộc cùng một lớp (trong trường hợp này là cùng một người).
# Embedding trong ngữ cảnh này là một vector số học đại diện cho một hình ảnh khuôn mặt trong không gian đa chiều

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import time
import h5py
import math
from tensorflow.python.platform import gfile
from six import iteritems # Khởi tạo embedding

def main(args):
    dataset = facenet.get_dataset(args.dataset_dir) # đọc thư mục dataset
    # Hàm này sẽ quét qua tất cả các thư mục con trong dataset_dir, mỗi thư mục con đại diện cho một lớp (trong trường hợp này là một người).
  
    # Sử dụng Pretrained model 20180402-114759.pb 
    with tf.Graph().as_default(): # Kết quả trả về (dataset) sẽ là một cấu trúc dữ liệu chứa thông tin về tất cả các lớp (người) và đường dẫn đến các hình ảnh tương ứng.
      
        # Đọc và xử lý dữ liệu ảnh
        image_list, label_list = facenet.get_image_paths_and_labels(dataset)
        nrof_images = len(image_list)
        image_indices = range(nrof_images)

        # Tải mô hình đã được huấn luyện
        image_batch, label_batch = facenet.read_and_augment_data(image_list,
            image_indices, args.image_size, args.batch_size, None, 
            False, False, False, nrof_preprocess_threads=4, shuffle=False)
        model_exp = os.path.expanduser(args.model_file)
        with gfile.FastGFile(model_exp,'rb') as f:
            graph_def = tf.GraphDef()

            #Lấy tensor embeddings
            graph_def.ParseFromString(f.read())
            input_map={'input':image_batch, 'phase_train':False}

            #Tạo session TensorFlow và thực hiện tính toán
            tf.import_graph_def(graph_def, input_map=input_map, name='net')

        embeddings = tf.get_default_graph().get_tensor_by_name("net/embeddings:0")

        with tf.Session() as sess:
            tf.train.start_queue_runners(sess=sess)
                
            embedding_size = int(embeddings.get_shape()[1])
            nrof_batches = int(math.ceil(nrof_images / args.batch_size))
            nrof_classes = len(dataset)
            label_array = np.array(label_list)
            class_names = [cls.name for cls in dataset]
            nrof_examples_per_class = [ len(cls.image_paths) for cls in dataset ]
            class_variance = np.zeros((nrof_classes,))

            # Tính toán trung tâm
            class_center = np.zeros((nrof_classes,embedding_size))
            distance_to_center = np.ones((len(label_list),))*np.NaN
            emb_array = np.zeros((0,embedding_size))

            # tính khoảng cách từ mỗi embedding đến trung tâm
            idx_array = np.zeros((0,), dtype=np.int32)
            lab_array = np.zeros((0,), dtype=np.int32)
            index_arr = np.append(0, np.cumsum(nrof_examples_per_class))
            for i in range(nrof_batches):
                t = time.time()
                emb, idx = sess.run([embeddings, label_batch])
                emb_array = np.append(emb_array, emb, axis=0)
                idx_array = np.append(idx_array, idx, axis=0)
                lab_array = np.append(lab_array, label_array[idx], axis=0)
                for cls in set(lab_array):
                    cls_idx = np.where(lab_array==cls)[0]
                    if cls_idx.shape[0]==nrof_examples_per_class[cls]:
                        # We have calculated all the embeddings for this class
                        i2 = np.argsort(idx_array[cls_idx])
                        emb_class = emb_array[cls_idx,:]
                        emb_sort = emb_class[i2,:] # Chứa tất cả các embedding của một lớp
                        center = np.mean(emb_sort, axis=0) # Tính trung bình theo cột, tạo ra một vector trung tâm.
                        diffs = emb_sort - center # Tính toán khoảng cách giữa từng embedding và trung tâm
                        dists_sqr = np.sum(np.square(diffs), axis=1) # Tính toán khoảng cách bình phương
                        class_variance[cls] = np.mean(dists_sqr) # Tính toán phương sai của lớp
                        class_center[cls,:] = center # Lưu trữ trung tâm của lớp
                        distance_to_center[index_arr[cls]:index_arr[cls+1]] = np.sqrt(dists_sqr) # Lưu trữ khoảng cách đến trung tâm
                        emb_array = np.delete(emb_array, cls_idx, axis=0) # Xóa các embedding của lớp đã được tính toán
                        idx_array = np.delete(idx_array, cls_idx, axis=0)
                        lab_array = np.delete(lab_array, cls_idx, axis=0)

                        
                print('Batch %d in %.3f seconds' % (i, time.time()-t))

            # Ghi dữ liệu lọc vào file
            print('Writing filtering data to %s' % args.data_file_name)
            mdict = {'class_names':class_names, 'image_list':image_list, 'label_list':label_list, 'distance_to_center':distance_to_center }
            with h5py.File(args.data_file_name, 'w') as f:
                for key, value in iteritems(mdict):
                    f.create_dataset(key, data=value)

# Định nghĩa các đối số dòng lệnh                     
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('dataset_dir', type=str,
        help='Path to the directory containing aligned dataset.')
    parser.add_argument('model_file', type=str,
        help='File containing the frozen model in protobuf (.pb) format to use for feature extraction.')
    parser.add_argument('data_file_name', type=str,
        help='The name of the file to store filtering data in.')
    parser.add_argument('--image_size', type=int,
        help='Image size.', default=160)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    return parser.parse_args(argv)

# Điểm vào chương trình
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
