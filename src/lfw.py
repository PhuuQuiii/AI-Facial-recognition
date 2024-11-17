# Đoạn mã này là một trợ giúp cho việc đánh giá trên tập dữ liệu "Labeled Faces in the Wild" (LFW). Nó chứa một số hàm để xử lý và đánh giá hiệu suất của mô hình nhận dạng khuôn mặt. Dưới đây là phần giải thích chi tiết cho từng phần trong mã:

# 3. Hàm `evaluate`:
#    - Nhận đầu vào là `embeddings` (các vector đặc trưng của khuôn mặt), `actual_issame` (một mảng cho biết các cặp khuôn mặt có phải là của cùng một người hay không), `nrof_folds`, `distance_metric`, và `subtract_mean`.
#    - Hàm này tính toán các chỉ số đánh giá như True Positive Rate (TPR), False Positive Rate (FPR), và độ chính xác của mô hình.
#    - Nó sử dụng hai hàm từ thư viện `facenet`: `calculate_roc` để tính TPR và FPR, và `calculate_val` để tính giá trị ngưỡng (val) và độ lệch chuẩn của nó (val_std).

# 4. Hàm `get_paths`:
#    - Nhận vào thư mục chứa LFW (`lfw_dir`) và các cặp hình ảnh (`pairs`).
#    - Tạo danh sách đường dẫn đến các tệp hình ảnh và một danh sách cho biết cặp hình ảnh có phải là cùng một người hay không (`issame_list`).
#    - Nếu cả hai hình ảnh trong cặp tồn tại, chúng sẽ được thêm vào danh sách. Nếu không, số lượng cặp bị bỏ qua sẽ được tăng lên và thông báo số lượng cặp bị bỏ qua sẽ được in ra.

# 5. Hàm `add_extension`:
#    - Nhận vào một đường dẫn và kiểm tra xem tệp với phần mở rộng `.jpg` hoặc `.png` có tồn tại hay không.
#    - Nếu tìm thấy tệp, trả về đường dẫn có phần mở rộng. Nếu không, sẽ phát sinh lỗi với thông báo rằng không tìm thấy tệp với phần mở rộng tương ứng.

# 6. Hàm `read_pairs`:
#    - Nhận vào tên tệp chứa danh sách các cặp hình ảnh.
#    - Đọc nội dung của tệp và tạo ra danh sách các cặp (mỗi cặp là một mảng chứa thông tin về hình ảnh).
#    - Trả về danh sách các cặp dưới dạng mảng NumPy.

# Mã này rất hữu ích trong việc đánh giá các mô hình nhận dạng khuôn mặt, đặc biệt là khi làm việc với tập dữ liệu LFW. Nó giúp người dùng đọc và xử lý dữ liệu đầu vào, tính toán các chỉ số đánh giá hiệu suất và tạo danh sách các cặp hình ảnh cho việc so sánh và đánh giá.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import facenet

def evaluate(embeddings, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = facenet.calculate_roc(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = facenet.calculate_val(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    return tpr, fpr, accuracy, val, val_std, far

def get_paths(lfw_dir, pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
            issame = True
        elif len(pair) == 4:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            path_list += (path0,path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs>0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)
    
    return path_list, issame_list
  
def add_extension(path):
    if os.path.exists(path+'.jpg'):
        return path+'.jpg'
    elif os.path.exists(path+'.png'):
        return path+'.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)

def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)



