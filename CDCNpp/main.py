from __future__ import print_function, division # For compatibility if needed

import cv2
import torch
from torchvision import transforms
import time
import threading
from queue import Queue, Empty # Thêm Empty để xử lý non-blocking get
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Attempt to import CDCNpp and Conv2d_cd.
# Ensure that your 'models' directory is in PYTHONPATH or in the same directory,
# and that CDCNs.py contains these definitions.
try:
    from .models.CDCNs import CDCNpp, Conv2d_cd # Assuming Conv2d_cd might be used by CDCNpp
except ImportError:
    print("LỖI: Không thể import CDCNpp từ models.CDCNs.")
    print("Hãy đảm bảo rằng file 'models/CDCNs.py' tồn tại và đúng cấu trúc,")
    print("hoặc thư mục 'models' có file '__init__.py' nếu CDCNs là một module.")
    print("Chương trình có thể không hoạt động chính xác nếu không có model này.")
    # Define dummy classes if real ones are not available so the script can be parsed
    if 'CDCNpp' not in globals():
        class CDCNpp(torch.nn.Module):
            def __init__(self, basic_conv=None, theta=0.7): # Added basic_conv and theta
                super(CDCNpp, self).__init__()
                # This is a DUMMY model. Replace with your actual model definition or ensure import works.
                self.conv1 = torch.nn.Conv2d(3, 1, kernel_size=3, padding=1)
                print("CẢNH BÁO: Sử dụng định nghĩa CDCNpp DUMMY. Model thực tế không được tải.")
            def forward(self, x):
                # Dummy forward returns a tensor of the same shape as input, but with 1 channel.
                # This is highly unlikely to match your actual model's output structure.
                # The actual CDCNpp returns multiple outputs, with map_x being the first.
                map_x = self.conv1(x)
                return map_x, None, None, None, None, None # Mimic tuple output
    if 'Conv2d_cd' not in globals() and 'basic_conv' not in locals(): # Check if basic_conv was actually needed
         class Conv2d_cd(torch.nn.Module): # Dummy
            def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, theta=0.0):
                super(Conv2d_cd, self).__init__()
                self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
            def forward(self, x):
                return self.conv(x)


# Global variable for prediction (optional, main function handles display label)
# latest_prediction_label = "Initializing..."
# prediction_lock = threading.Lock()

def load_model(model_path, default_theta=0.7):
    """
    Tải model CDCNpp đã được huấn luyện.
    default_theta: Giá trị theta mặc định để khởi tạo CDCNpp,
                   phải khớp với giá trị được sử dụng trong train_test.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): # For MacOS Metal
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    model = None
    try:
        # Cố gắng khởi tạo model với theta.
        # Giả định rằng CDCNpp được import thành công và chấp nhận tham số theta.
        # Và Conv2d_cd cũng được import/định nghĩa nếu CDCNpp cần nó.
        model = CDCNpp(basic_conv=Conv2d_cd, theta=default_theta).to(device)
    except Exception as e:
        print(f"Lỗi khi khởi tạo CDCNpp với theta: {e}. Thử khởi tạo mặc định.")
        try:
            model = CDCNpp().to(device) # Thử khởi tạo không có theta nếu ở trên thất bại
        except Exception as e_inner:
            print(f"LỖI: Không thể khởi tạo model CDCNpp: {e_inner}")
            return None, None

    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Model weights loaded successfully from {model_path}")
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file model tại: {model_path}")
        print("Vui lòng kiểm tra lại đường dẫn và tên file.")
        return None, None
    except Exception as e:
        print(f"LỖI khi tải model state_dict: {e}")
        return None, None
    model.eval()
    return model, device

def preprocess(frame, device):
    """
    Tiền xử lý frame đầu vào.
    Đảm bảo các bước này (đặc biệt là Normalization) khớp với quá trình training/validation.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    transform_ops = [
        transforms.ToPILImage(),
        transforms.Resize((256, 256)), # Kích thước mong đợi của model
        transforms.ToTensor(),
        # QUAN TRỌNG: Thêm Normalization nếu model của bạn được huấn luyện với nó.
        # Các giá trị mean/std phải khớp với những gì được sử dụng trong Normaliztion_valtest.
        # Ví dụ:
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # Hoặc các giá trị cụ thể mà Normaliztion_valtest của bạn đã sử dụng.
    ]
    
    transform = transforms.Compose(transform_ops)
    return transform(frame_rgb).unsqueeze(0).to(device)

def predict_on_worker(model, frame_to_predict, device, threshold=0.9):
    """
    Thực hiện tiền xử lý và dự đoán trên một frame,
    với logic tính điểm được điều chỉnh để giống với train_test.
    LƯU Ý: `threshold` có thể cần điều chỉnh dựa trên phân phối điểm mới.
    """
    if frame_to_predict is None:
        # print("Error: Input frame is None in predict_on_worker.") # Có thể quá nhiều log
        return "Error: No frame"
    if model is None:
        # print("Error: Model is None in predict_on_worker.")
        return "Error: No model"

    try:
        input_tensor = preprocess(frame_to_predict, device)
    except Exception as e:
        print(f"Lỗi trong quá trình tiền xử lý: {e}")
        return "Error: Preprocessing"

    # model.eval() # Đã được set trong load_model và worker thread không thay đổi nó
    with torch.no_grad():
        try:
            output_from_model = model(input_tensor)
        except Exception as e:
            print(f"Lỗi khi thực thi model: {e}")
            return "Error: Model exec", 0.0

        # Trích xuất map_x (giả định là output đầu tiên, như trong train_test)
        if isinstance(output_from_model, (tuple, list)):
            if not output_from_model: # List/tuple rỗng
                print("Lỗi: Model output rỗng.")
                return "Error: Empty output", 0.0
            map_x = output_from_model[0]
            if map_x is None: # Output đầu tiên là None
                print("Lỗi: map_x (output đầu tiên của model) là None.")
                return "Error: map_x is None", 0.0
        else:
            map_x = output_from_model # Nếu model trả về một tensor duy nhất

        if not isinstance(map_x, torch.Tensor):
            print(f"Lỗi: map_x không phải là Tensor (là {type(map_x)}).")
            return "Error: map_x type", 0.0

        # Tính điểm tương tự score_norm của train_test,
        # giả sử binary_mask là toàn 1 và bao phủ toàn bộ map_x.
        # score_norm = torch.sum(map_x) / torch.sum(binary_mask)
        # Trở thành torch.mean(map_x) nếu binary_mask là toàn 1.
        if map_x.numel() == 0:
            print("Lỗi: map_x không có phần tử nào (numel is 0).")
            return "Error: map_x empty", 0.0
            
        pred_score_unclipped = torch.mean(map_x).item()
        pred_score = max(0.0, min(1.0, pred_score_unclipped))
        
        # Áp dụng clipping, tương tự cách train_test clip map_score cuối cùng.
        pred_score = pred_score_unclipped
        if pred_score > 1.0:
            pred_score = 1.0
        # Tùy chọn: train_test không clip tại 0, nhưng nếu map_x có thể âm
        # và điểm số được kỳ vọng là không âm:
        elif pred_score < 0.0: # Thêm elif để tránh gán lại nếu đã > 1.0
             pred_score = 0.0
            
        print(f"Prediction score (mean of map_x, clipped): {pred_score:.4f} {'Real' if pred_score > threshold else 'Fake'}")
        label = "Real" if pred_score > threshold else "Fake"
        return label, pred_score  # <-- trả về cả nhãn và xác suất

def prediction_thread_worker(model, device, frame_input_queue, result_output_queue, decision_threshold):
    """Luồng worker nhận frame, dự đoán và đặt kết quả vào result_output_queue."""
    print("Prediction worker thread started.")
    while True:
        try:
            frame_to_predict = frame_input_queue.get(timeout=1) # Chờ tối đa 1 giây
        except Empty:
            continue # Nếu queue rỗng, tiếp tục vòng lặp

        if frame_to_predict is None: # Tín hiệu dừng thread
            print("Prediction worker thread received stop signal.")
            result_output_queue.put(None) # Báo cho main thread
            break

        if model is None or device is None:
            # print("Prediction worker: Model hoặc device chưa được khởi tạo.") # Có thể quá nhiều log
            result_output_queue.put("Error: Model/Device NA") # Gửi lỗi nếu có
            continue

        prediction_result = predict_on_worker(model, frame_to_predict, device, threshold=decision_threshold)
        
        # Cố gắng đặt kết quả vào result_output_queue không block
        try:
            result_output_queue.put_nowait(prediction_result)
        except Exception: # queue.Full
             # Nếu queue đầy, bỏ qua kết quả này để không làm main thread bị block
             # Hoặc có thể xóa item cũ và thêm item mới nếu muốn luôn có kết quả mới nhất
            try:
                result_output_queue.get_nowait() # Xóa item cũ
            except Empty:
                pass
            try:
                result_output_queue.put_nowait(prediction_result) # Thử lại
            except Exception:
                pass # Nếu vẫn đầy thì bỏ qua

        frame_input_queue.task_done()
    print("Prediction worker thread finished.")


def main():
    # Đường dẫn tới model của bạn - THAY ĐỔI NẾU CẦN
    model_path = r"D:\E\DoANChuyenNganh\Facial_recognition\CDCNpp\CDCNpp_BinaryMask_P1_07_60.pkl" # Sử dụng đường dẫn tương đối hoặc tuyệt đối
    # model_path = r"/Users/bduong/Documents/CDCN/FAS_challenge_CVPRW2020/Track2_Single-modal/model1_pytorch/CDCNpp_BinaryMask_P1_07/CDCNpp_BinaryMask_P1_07_30.pkl"

    # Giá trị theta được sử dụng khi huấn luyện model (từ args.theta của train_test, mặc định là 0.7)
    training_theta_value = 0.7
    # Ngưỡng quyết định Real/Fake. CÓ THỂ CẦN ĐIỀU CHỈNH sau khi thay đổi cách tính điểm.
    decision_threshold = 0.9 # Ví dụ: Điều chỉnh ngưỡng này dựa trên kết quả thực tế

    actual_model, device = load_model(model_path, default_theta=training_theta_value)

    if actual_model is None:
        print("Không thể khởi tạo model. Thoát chương trình.")
        return

    cap = cv2.VideoCapture(0) # Mở camera mặc định
    if not cap.isOpened():
        print("Không mở được camera!")
        return

    # Queues để giao tiếp giữa các luồng
    frame_predict_queue = Queue(maxsize=1) # Worker chỉ xử lý frame gần nhất
    result_predict_queue = Queue(maxsize=1) # Main thread chỉ nhận kết quả gần nhất

    worker_thread = threading.Thread(target=prediction_thread_worker,
                                     args=(actual_model, device, frame_predict_queue, result_predict_queue, decision_threshold),
                                     daemon=True)
    worker_thread.start()

    frame_count_for_trigger = 0
    prediction_interval = 5 # Dự đoán mỗi X frame (ví dụ: 5 frames ~ 100-200ms delay @ 25-50 FPS)

    fps_start_time = time.time()
    fps_frame_count = 0
    fps_display = "FPS: ??"
    current_label_to_display = "Initializing..."

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc frame từ camera. Kết thúc.")
            break

        # Tính toán FPS
        fps_frame_count += 1
        current_time = time.time()
        if (current_time - fps_start_time) >= 1.0:
            fps = fps_frame_count / (current_time - fps_start_time)
            fps_display = f"FPS: {fps:.2f}"
            fps_frame_count = 0
            fps_start_time = current_time

        frame_count_for_trigger += 1
        if frame_count_for_trigger >= prediction_interval:
            frame_count_for_trigger = 0
            try:
                # Tạo bản sao của frame để gửi đi, tránh race conditions
                # nếu frame gốc bị thay đổi trước khi worker xử lý.
                frame_copy = frame.copy()
                frame_predict_queue.put_nowait(frame_copy)
            except Exception: # queue.Full (tên exception là queue.Full, nhưng bắt Exception chung cho an toàn)
                pass # Bỏ qua frame này nếu worker đang bận

        # Kiểm tra kết quả từ worker (không block)
        try:
            new_prediction = result_predict_queue.get_nowait()
            if new_prediction is None: # Tín hiệu worker đã dừng
                print("Main thread: Worker stopped processing.")
                break
            current_label_to_display = new_prediction
            # result_predict_queue.task_done() # Không cần task_done() cho get_nowait() nếu không join queue
        except Empty:
            pass # Không có kết quả mới

        # Hiển thị
        label_color = (0, 255, 255) # Vàng cho initializing/error
        if current_label_to_display == "Real":
            label_color = (0, 255, 0) # Xanh lá
        elif current_label_to_display == "Fake":
            label_color = (0, 0, 255) # Đỏ
        
        cv2.putText(frame, str(current_label_to_display), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2)
        cv2.putText(frame, fps_display, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow('Face Anti-Spoofing (CDCNpp)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Phím 'q' được nhấn, đang thoát...")
            break

    print("Đang dừng chương trình...")
    frame_predict_queue.put(None) # Gửi tín hiệu dừng cho worker

    if worker_thread.is_alive():
        print("Đang chờ luồng worker kết thúc...")
        worker_thread.join(timeout=5) # Đợi tối đa 5 giây
        if worker_thread.is_alive():
            print("CẢNH BÁO: Luồng worker không kết thúc kịp thời.")

    cap.release()
    cv2.destroyAllWindows()
    print("Chương trình đã kết thúc.")

if __name__ == "__main__":
    main()