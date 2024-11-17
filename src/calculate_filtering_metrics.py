
    #    1.     Xử lý dữ liệu khuôn mặt    
    #       `dataset = facenet.get_dataset(args.dataset_dir)`    : Đọc tập dữ liệu khuôn mặt từ thư mục `dataset_dir`. Mỗi thư mục con trong `dataset_dir` đại diện cho một lớp (một người) và chứa các ảnh khuôn mặt của người đó. Kết quả `dataset` chứa thông tin về các lớp và đường dẫn đến ảnh tương ứng.
    #       Mô hình đã huấn luyện trước (pretrained model)    : Script này sử dụng mô hình được huấn luyện trước (`20180402  114759.pb`) để tạo ra các `embedding` cho mỗi ảnh khuôn mặt.
 

    #    2.     Xử lý ảnh và tạo Embedding    
    #       Đọc và xử lý dữ liệu ảnh    : Các ảnh được lấy từ `image_list`, là danh sách đường dẫn đến ảnh, và `label_list`, là danh sách nhãn cho các ảnh tương ứng. `nrof_images` là tổng số ảnh trong tập dữ liệu.
    #       `facenet.read_and_augment_data(...)`    : Hàm này tạo các batch ảnh (theo kích thước `batch_size`) và áp dụng tiền xử lý cho ảnh để chuẩn bị cho quá trình tạo `embedding`.
    #       Tải mô hình và khởi tạo TensorFlow session    : Script nạp file mô hình `model_file` (.pb) và thiết lập các tensor cần thiết. Tensor `embeddings` đại diện cho các `embedding` của ảnh trong không gian đa chiều.
     

    #    3.     Tính toán trung tâm và phương sai    
    #       `embedding_size`    : Kích thước của vector `embedding`.
    #       `class_center` và `distance_to_center`    : `class_center` lưu trữ trung tâm của từng lớp, còn `distance_to_center` lưu khoảng cách từ mỗi `embedding` đến trung tâm lớp tương ứng.
    #       `emb_array`    : Lưu trữ các `embedding` của ảnh, còn `lab_array` chứa nhãn tương ứng.
    #       Tính trung tâm cho từng lớp    :
    #     Lặp qua các batch ảnh và tính trung tâm cho mỗi lớp bằng cách tính giá trị trung bình của các `embedding` thuộc cùng lớp.
    #         Tính khoảng cách giữa từng embedding và trung tâm lớp    : Đo khoảng cách Euclidean bình phương và tính phương sai (`class_variance`) cho mỗi lớp.
   

    #    4.     Lưu trữ kết quả    
    #       `distance_to_center`    : Lưu khoảng cách của mỗi `embedding` đến trung tâm lớp của nó.
    #       Ghi dữ liệu lọc vào file HDF5    : Lưu các thông tin quan trọng (như tên lớp, danh sách ảnh, danh sách nhãn, và khoảng cách đến trung tâm) vào file `data_file_name`. Điều này giúp lưu trữ và truy xuất dữ liệu dễ dàng.
 

    #    5.     Tiền xử lý dữ liệu    
    #   Script này có thể được dùng như một công cụ tiền xử lý để cải thiện chất lượng tập dữ liệu khuôn mặt trước khi thực hiện nhận dạng. Những ảnh có khoảng cách đến trung tâm quá lớn có thể bị coi là chất lượng kém hoặc không phù hợp.

    #    6.     Định nghĩa các đối số dòng lệnh    
    #   Sử dụng `argparse` để xác định các tham số của script, bao gồm:
    #     `dataset_dir`: Đường dẫn đến thư mục chứa tập dữ liệu.
    #     `model_file`: Đường dẫn đến file mô hình đã huấn luyện trước.
    #     `data_file_name`: Tên file để lưu trữ kết quả lọc.
    #     `image_size` và `batch_size`: Kích thước ảnh và số lượng ảnh trong mỗi batch.

      

      
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
