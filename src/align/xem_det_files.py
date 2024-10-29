import numpy as np

# Xem nội dung của det1, det2, và det3
# det1, det2, và det3 có thể tương ứng với ba giai đoạn khác nhau của mạng MTCNN


# 1. Đây là các file NumPy (.npy) chứa dữ liệu được sử dụng trong mô hình MTCNN (Multi-task Cascaded Convolutional Networks) để phát hiện khuôn mặt.

# 2. Mỗi file tương ứng với một giai đoạn khác nhau trong quá trình phát hiện khuôn mặt của MTCNN:

#    - det1.npy: Chứa tham số cho giai đoạn đầu tiên (P-Net), thường là một mạng CNN nhỏ để nhanh chóng tìm ra các vùng có khả năng chứa khuôn mặt.
   
#    - det2.npy: Chứa tham số cho giai đoạn thứ hai (R-Net), một mạng phức tạp hơn để tinh chỉnh các vùng đã phát hiện.
   
#    - det3.npy: Chứa tham số cho giai đoạn cuối cùng (O-Net), mạng phức tạp nhất để xác định chính xác vị trí khuôn mặt và các điểm đặc trưng.

# 3. Các file này chứa các trọng số (weights) và độ lệch (biases) đã được huấn luyện trước cho mỗi giai đoạn của mạng MTCNN.

# 4. Trong quá trình phát hiện khuôn mặt, các tham số này được tải vào mô hình để thực hiện các phép tính cần thiết.

# 5. Việc sử dụng các file .npy này giúp tăng tốc quá trình khởi tạo mô hình và đảm bảo tính nhất quán của kết quả phát hiện khuôn mặt.

def load_and_print_info(file_name):
    # Thêm encoding='latin1' để giải quyết lỗi UnicodeDecodeError
    data = np.load(file_name, allow_pickle=True, encoding='latin1')
    
    print(f"\nThông tin về file {file_name}:")
    print(f"Kiểu dữ liệu: {type(data)}")
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
    
    if isinstance(data, np.ndarray):
        print("Một số phần tử đầu tiên:")
        print(data.flatten()[:10])
    elif isinstance(data, dict):
        print("Các khóa trong dictionary:")
        for key in data.keys():
            print(f"- {key}: {type(data[key])}")
            if isinstance(data[key], np.ndarray):
                print(f"  Shape: {data[key].shape}")
                print(f"  Một số phần tử đầu tiên: {data[key].flatten()[:5]}")
    else:
        print("Không thể hiển thị nội dung chi tiết.")

# Danh sách các file cần kiểm tra
files = ['det1.npy', 'det2.npy', 'det3.npy']

for file in files:
    load_and_print_info(file)


# Trọng số (weights) và độ lệch (biases) là hai thành phần quan trọng trong mạng neural như MTCNN.

# 1. Trọng số (Weights):
#    - Là các tham số được học trong quá trình huấn luyện mạng neural.
#    - Đại diện cho mức độ quan trọng của các kết nối giữa các neuron trong mạng.
#    - Trong các lớp tích chập (convolutional layers), trọng số là các bộ lọc (filters) được sử dụng để trích xuất đặc trưng từ dữ liệu đầu vào.
#    - Trọng số được điều chỉnh trong quá trình huấn luyện để tối ưu hóa hiệu suất của mạng.

# 2. Độ lệch (Biases):
#    - Là các tham số bổ sung được thêm vào sau khi áp dụng trọng số.
#    - Cho phép mạng neural dịch chuyển hàm kích hoạt sang trái hoặc phải, giúp tăng tính linh hoạt của mô hình.
#    - Giúp mạng neural học được các mẫu phức tạp hơn bằng cách thêm một giá trị không đổi vào đầu ra của mỗi neuron.

# Trong ngữ cảnh của MTCNN:
# - Mỗi giai đoạn (P-Net, R-Net, O-Net) có các bộ trọng số và độ lệch riêng.
# - Các file det1.npy, det2.npy, và det3.npy chứa các giá trị này cho mỗi giai đoạn tương ứng.
# - Khi mạng được khởi tạo, nó sẽ tải các giá trị này từ các file .npy và sử dụng chúng trong quá trình xử lý hình ảnh để phát hiện khuôn mặt.

# Trong file detect_face.py, có thể thấy việc sử dụng trọng số và độ lệch trong các phương thức như conv() và fc():


