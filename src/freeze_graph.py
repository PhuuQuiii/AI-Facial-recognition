# Đoạn mã này được viết bằng Python và sử dụng TensorFlow để xuất một mô hình học sâu thành một định dạng có thể lưu trữ và sử dụng lại. Dưới đây là giải thích chi tiết từng phần trong đoạn mã:

#  2. Hàm `main(args)`
# - Tạo một đồ thị TensorFlow mới và bắt đầu một phiên làm việc (`Session`).
# - Tải mô hình bằng cách sử dụng tên tệp metagraph và checkpoint từ thư mục mô hình mà người dùng đã chỉ định.
# - In ra đường dẫn đến các tệp metagraph và checkpoint.
# - Khởi tạo các biến toàn cục và cục bộ trong mô hình.
# - Khôi phục mô hình từ checkpoint.
# - Lấy định nghĩa đồ thị và sửa đổi các nút chuẩn hóa (batch norm).
# - Gọi hàm `freeze_graph_def` để chuyển đổi các biến trong đồ thị thành hằng số.

#  3. Hàm `freeze_graph_def(sess, input_graph_def, output_node_names)`
# - Sửa đổi các nút trong định nghĩa đồ thị để thay thế một số loại nút (như `RefSwitch`, `AssignSub`, `AssignAdd`) bằng các nút tương ứng mà không yêu cầu khoá (lock).
# - Tạo danh sách các nút quan trọng cần giữ lại trong đồ thị (như các nút liên quan đến đầu vào và đầu ra của mô hình).
# - Sử dụng `graph_util.convert_variables_to_constants` để thay thế tất cả các biến trong đồ thị bằng các hằng số với cùng giá trị, chỉ định các nút đầu ra để giữ lại.

#  4. Hàm `parse_arguments(argv)`
# - Định nghĩa và phân tích các tham số đầu vào từ dòng lệnh.
# - Chấp nhận hai đối số: thư mục chứa mô hình và tên tệp xuất ra định dạng protobuf.

# Đoạn mã này giúp chuyển đổi một mô hình học sâu (cụ thể là một mô hình nhận diện khuôn mặt) thành một định dạng có thể sử dụng lại mà không cần phải khởi tạo lại các biến.
# Nó hỗ trợ việc đóng gói mô hình và chia sẻ mô hình đã huấn luyện cho các ứng dụng khác mà không cần phải biết đến kiến trúc bên trong hoặc các tham số.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import graph_util
import tensorflow as tf
import argparse
import os
import sys
import facenet
from six.moves import xrange  # @UnresolvedImport

def main(args):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model metagraph and checkpoint
            print('Model directory: %s' % args.model_dir)
            meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(args.model_dir))
            
            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)

            model_dir_exp = os.path.expanduser(args.model_dir)
            saver = tf.train.import_meta_graph(os.path.join(model_dir_exp, meta_file), clear_devices=True)
            tf.get_default_session().run(tf.global_variables_initializer())
            tf.get_default_session().run(tf.local_variables_initializer())
            saver.restore(tf.get_default_session(), os.path.join(model_dir_exp, ckpt_file))
            
            # Retrieve the protobuf graph definition and fix the batch norm nodes
            input_graph_def = sess.graph.as_graph_def()
            
            # Freeze the graph def
            output_graph_def = freeze_graph_def(sess, input_graph_def, 'embeddings,label_batch')

        # Serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(args.output_file, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph: %s" % (len(output_graph_def.node), args.output_file))
        
def freeze_graph_def(sess, input_graph_def, output_node_names):
    for node in input_graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in xrange(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr: del node.attr['use_locking']
    
    # Get the list of important nodes
    whitelist_names = []
    for node in input_graph_def.node:
        if (node.name.startswith('InceptionResnet') or node.name.startswith('embeddings') or 
                node.name.startswith('image_batch') or node.name.startswith('label_batch') or
                node.name.startswith('phase_train') or node.name.startswith('Logits')):
            whitelist_names.append(node.name)

    # Replace all the variables in the graph with constants of the same values
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, input_graph_def, output_node_names.split(","),
        variable_names_whitelist=whitelist_names)
    return output_graph_def
  
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model_dir', type=str, 
        help='Directory containing the metagraph (.meta) file and the checkpoint (ckpt) file containing model parameters')
    parser.add_argument('output_file', type=str, 
        help='Filename for the exported graphdef protobuf (.pb)')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
