# Đoạn mã trên là một script để giải mã và xử lý dữ liệu hình ảnh từ tập dữ liệu MsCelebV1 được lưu dưới định dạng TSV (Tab Separated Values).
# Dữ liệu TSV này chứa các thông tin về hình ảnh của người nổi tiếng, được mã hóa dưới dạng base64. 

#  2. Cấu trúc của tập tin TSV:
# - Tập tin TSV có sáu cột, mỗi cột chứa các thông tin sau:
#   - Cột 1: `Freebase MID` - ID của người nổi tiếng trong cơ sở dữ liệu Freebase.
#   - Cột 2: `Query/Name` - Tên hoặc truy vấn của người nổi tiếng.
#   - Cột 3: `ImageSearchRank` - Độ xếp hạng của hình ảnh.
#   - Cột 4: `ImageURL` - URL của hình ảnh.
#   - Cột 5: `PageURL` - URL của trang chứa hình ảnh.
#   - Cột 6: `ImageData_Base64Encoded` - Dữ liệu hình ảnh mã hóa dưới dạng base64.

#  3. Hàm `main(args)`:
# Hàm `main` là hàm chính, có chức năng:
# - Tạo thư mục đầu ra: Từ tham số `output_dir`, kiểm tra xem thư mục đã tồn tại chưa, nếu chưa thì tạo mới thư mục này.
# - Lưu thông tin phiên bản mã nguồn: Dùng `facenet.store_revision_info` để lưu thông tin mã nguồn của phiên bản hiện tại trong thư mục đầu ra.
# - Đọc và giải mã dữ liệu hình ảnh: Vòng lặp duyệt qua từng tập tin TSV trong `args.tsv_files`, với từng dòng:
#   - Tách dòng thành các `fields` bằng cách sử dụng dấu tab (`\t`) làm ký tự phân tách.
#   - Tạo thư mục `class_dir` dựa trên `Freebase MID`.
#   - Tạo tên hình ảnh `img_name` bằng cách ghép `Query/Name` và `PageURL`.
#   - Giải mã chuỗi base64 trong `fields[5]` để thu được dữ liệu hình ảnh (`img_data`), sau đó chuyển đổi dữ liệu này thành mảng ảnh `img` bằng OpenCV (`cv2`).
#   - Nếu có tham số `size`, ảnh sẽ được thay đổi kích thước về kích thước đã cho (`args.size`).
#   - Lưu ảnh vào thư mục `class_dir` với định dạng `png` hoặc `jpg` như đã xác định trong `args.output_format`.

#  4. Hàm `argparse.ArgumentParser()`:
# - Định nghĩa các tham số dòng lệnh mà script nhận vào, gồm:
#   - `output_dir`: Thư mục đầu ra cho tập dữ liệu ảnh.
#   - `tsv_files`: Danh sách các tập tin TSV đầu vào.
#   - `--size`: Tham số tùy chọn để xác định kích thước ảnh đầu ra.
#   - `--output_format`: Định dạng ảnh đầu ra, mặc định là `png`.
 
# Đoạn mã này thực hiện việc giải mã và lưu ảnh từ tập dữ liệu MsCelebV1 theo các bước sau:
# 1. Đọc dữ liệu TSV.
# 2. Giải mã dữ liệu ảnh từ base64.
# 3. Thay đổi kích thước ảnh (nếu cần) và lưu vào thư mục đầu ra theo cấu trúc `Freebase MID`.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import numpy as np
import base64
import sys
import os
import cv2
import argparse
import facenet


# File format: text files, each line is an image record containing 6 columns, delimited by TAB.
# Column1: Freebase MID
# Column2: Query/Name
# Column3: ImageSearchRank
# Column4: ImageURL
# Column5: PageURL
# Column6: ImageData_Base64Encoded

def main(args):
    output_dir = os.path.expanduser(args.output_dir)
  
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
  
    # Store some git revision info in a text file in the output directory
    src_path,_ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
    
    i = 0
    for f in args.tsv_files:
        for line in f:
            fields = line.split('\t')
            class_dir = fields[0]
            img_name = fields[1] + '-' + fields[4] + '.' + args.output_format
            img_string = fields[5]
            img_dec_string = base64.b64decode(img_string)
            img_data = np.fromstring(img_dec_string, dtype=np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR) #pylint: disable=maybe-no-member
            if args.size:
                img = misc.imresize(img, (args.size, args.size), interp='bilinear')
            full_class_dir = os.path.join(output_dir, class_dir)
            if not os.path.exists(full_class_dir):
                os.mkdir(full_class_dir)
            full_path = os.path.join(full_class_dir, img_name.replace('/','_'))
            cv2.imwrite(full_path, img) #pylint: disable=maybe-no-member
            print('%8d: %s' % (i, full_path))
            i += 1
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('output_dir', type=str, help='Output base directory for the image dataset')
    parser.add_argument('tsv_files', type=argparse.FileType('r'), nargs='+', help='Input TSV file name(s)')
    parser.add_argument('--size', type=int, help='Images are resized to the given size')
    parser.add_argument('--output_format', type=str, help='Format of the output images', default='png', choices=['png', 'jpg'])

    main(parser.parse_args())

