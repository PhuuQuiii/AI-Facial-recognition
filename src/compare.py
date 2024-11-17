# Đây là mã Python cho phép căn chỉnh khuôn mặt và tính toán khoảng cách L2 giữa các embeddings (biểu diễn đặc trưng)
#  của các ảnh khuôn mặt. Mã này sử dụng TensorFlow và thư viện `facenet` để tải mô hình học sâu và tính toán embeddings khuôn mặt,
#  cũng như thư viện `align.detect_face` để phát hiện khuôn mặt.

#  2. Hàm chính (`main`)

# - `main(args)` lấy danh sách ảnh đầu vào từ `args`, tải và căn chỉnh ảnh qua hàm `load_and_align_data`.
# - `tf.Graph().as_default()` tạo đồ thị tính toán của TensorFlow.
# - `facenet.load_model(args.model)` tải mô hình đã được huấn luyện trước.
# - `images_placeholder`, `embeddings`, và `phase_train_placeholder` là các tensor đầu vào và đầu ra của mô hình.
# - `feed_dict` truyền các ảnh đã căn chỉnh vào mô hình để tính embeddings.
# - Tính toán ma trận khoảng cách giữa các ảnh bằng cách tính khoảng cách L2 giữa các embeddings.

#  3. Hàm `load_and_align_data`
# - `minsize`, `threshold`, và `factor` xác định các tham số để phát hiện khuôn mặt.
# - `pnet`, `rnet`, và `onet` là các mạng dùng trong MTCNN để phát hiện khuôn mặt.
# - Duyệt qua danh sách ảnh, mỗi ảnh sẽ:
#   - Được đọc và chuyển về RGB.
#   - Xác định hộp chứa khuôn mặt (`bounding_boxes`) bằng MTCNN.
#   - Cắt và căn chỉnh khuôn mặt theo hộp đã phát hiện và `margin` (lề).
#   - Chuẩn hóa khuôn mặt (`prewhitened`) trước khi thêm vào `img_list`.
# - Trả về mảng `images` chứa các khuôn mặt đã căn chỉnh.

#  4. Hàm `parse_arguments`
# - Đọc các tham số từ dòng lệnh, bao gồm:
#   - `model`: đường dẫn tới mô hình đã huấn luyện.
#   - `image_files`: danh sách đường dẫn ảnh cần so sánh.
#   - `image_size`, `margin`, `gpu_memory_fraction`: các tham số tùy chọn.

# Mã này đọc và căn chỉnh các ảnh khuôn mặt, sau đó sử dụng mô hình `facenet` để tạo embeddings.
#  Nó tính toán và in ra ma trận khoảng cách giữa các embeddings của các ảnh,
#  từ đó đánh giá mức độ tương đồng giữa các khuôn mặt.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import numpy as np
import sys
import os
import copy
import argparse
import facenet
import align.detect_face

def main(args):

    images = load_and_align_data(args.image_files, args.image_size, args.margin, args.gpu_memory_fraction)
    with tf.Graph().as_default():

        with tf.Session() as sess:
      
            # Load the model
            facenet.load_model(args.model)
    
            # Get input and output tensors
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)
            
            nrof_images = len(args.image_files)

            print('Images:')
            for i in range(nrof_images):
                print('%1d: %s' % (i, args.image_files[i]))
            print('')
            
            # Print distance matrix
            print('Distance matrix')
            print('    ', end='')
            for i in range(nrof_images):
                print('    %1d     ' % i, end='')
            print('')
            for i in range(nrof_images):
                print('%1d  ' % i, end='')
                for j in range(nrof_images):
                    dist = np.sqrt(np.sum(np.square(np.subtract(emb[i,:], emb[j,:]))))
                    print('  %1.4f  ' % dist, end='')
                print('')
            
            
def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
  
    tmp_image_paths=copy.copy(image_paths)
    img_list = []
    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
          image_paths.remove(image)
          print("can't detect face, remove ", image)
          continue
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
    images = np.stack(img_list)
    return images

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('image_files', type=str, nargs='+', help='Images to compare')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
