# Đoạn mã trên là một script Python được sử dụng để tải xuống và giải nén các mô hình đã được lưu trữ trên Google Drive. Đây là phần mềm tiện lợi cho việc quản lý và tải mô hình cho các ứng dụng máy học. Dưới đây là phần giải thích chi tiết về từng phần của mã:

#  1. Thư viện và Từ điển Mô hình:
# - `requests`: Thư viện dùng để gửi các yêu cầu HTTP, rất hữu ích trong việc tải xuống tệp từ internet.
# - `zipfile`: Thư viện dùng để làm việc với các tệp ZIP, cho phép giải nén các tệp nén.
# - `os`: Thư viện giúp tương tác với hệ thống tệp, như tạo thư mục hoặc kiểm tra sự tồn tại của tệp.


# - `model_dict`: Từ điển chứa tên mô hình và ID tương ứng của chúng trên Google Drive. ID này được sử dụng để tải mô hình cụ thể từ Drive.

#  2. Hàm `download_and_extract_file(model_name, data_dir)`:
# Hàm này thực hiện hai nhiệm vụ chính: tải xuống tệp ZIP chứa mô hình và giải nén tệp này vào thư mục được chỉ định.
# - `model_name`: Tên của mô hình cần tải xuống.
# - `data_dir`: Thư mục nơi tệp ZIP sẽ được lưu và giải nén.
# - Đầu tiên, hàm kiểm tra xem tệp đã tồn tại chưa. Nếu chưa, nó sẽ gọi hàm `download_file_from_google_drive` để tải tệp xuống và sau đó giải nén tệp đó.

#  3. Hàm `download_file_from_google_drive(file_id, destination)`:
# Hàm này thực hiện việc gửi yêu cầu đến Google Drive để tải tệp.

# - `file_id`: ID của tệp cần tải xuống.
# - `destination`: Đường dẫn nơi tệp sẽ được lưu.
# - Hàm này gửi yêu cầu GET đến Google Drive với ID tệp. Nếu có yêu cầu xác nhận (do Google Drive yêu cầu cho tệp lớn), hàm sẽ lấy mã xác nhận thông qua `get_confirm_token` và gửi yêu cầu lại với mã xác nhận.

#  4. Hàm `get_confirm_token(response)`:
# Hàm này lấy mã xác nhận từ phản hồi của Google Drive nếu cần.

# - Kiểm tra xem phản hồi có chứa cookie nào bắt đầu bằng `download_warning` hay không. Nếu có, nó sẽ trả về giá trị của cookie đó, giúp quá trình tải xuống diễn ra thành công.

#  5. Hàm `save_response_content(response, destination)`:
# Hàm này lưu nội dung của phản hồi vào tệp đã chỉ định.

# - `CHUNK_SIZE`: Kích thước từng phần (chunk) dữ liệu được tải xuống.
# - Mở tệp ở chế độ ghi nhị phân (`wb`) và ghi dữ liệu vào tệp từng phần một. Điều này giúp tiết kiệm bộ nhớ và xử lý các tệp lớn một cách hiệu quả.

# Đoạn mã này là một script hoàn chỉnh cho việc tải xuống và giải nén các mô hình từ Google Drive. Người dùng chỉ cần gọi hàm `download_and_extract_file` với tên mô hình và thư mục đích, và mã sẽ tự động tải mô hình xuống, giải nén và lưu vào thư mục đã chỉ định.

import requests
import zipfile
import os

model_dict = {
    'lfw-subset':      '1B5BQUZuJO-paxdN8UclxeHAR1WnR_Tzi', 
    '20170131-234652': '0B5MzpY9kBtDVSGM0RmVET2EwVEk',
    '20170216-091149': '0B5MzpY9kBtDVTGZjcWkzT3pldDA',
    '20170512-110547': '0B5MzpY9kBtDVZ2RpVDYwWmxoSUk',
    '20180402-114759': '1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-'
    }

def download_and_extract_file(model_name, data_dir):
    file_id = model_dict[model_name]
    destination = os.path.join(data_dir, model_name + '.zip')
    if not os.path.exists(destination):
        print('Downloading file to %s' % destination)
        download_file_from_google_drive(file_id, destination)
        with zipfile.ZipFile(destination, 'r') as zip_ref:
            print('Extracting file to %s' % data_dir)
            zip_ref.extractall(data_dir)

def download_file_from_google_drive(file_id, destination):
    
        URL = "https://drive.google.com/uc?export=download"
    
        session = requests.Session()
    
        response = session.get(URL, params = { 'id' : file_id }, stream = True)
        token = get_confirm_token(response)
    
        if token:
            params = { 'id' : file_id, 'confirm' : token }
            response = session.get(URL, params = params, stream = True)
    
        save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
