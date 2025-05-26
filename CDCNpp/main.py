import cv2
import torch
from torchvision import transforms
from .models.CDCNs import CDCNpp  # Sửa lại dòng import này
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# def detect_face(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
#     return len(faces) > 0

def load_model(model_path):
    model = CDCNpp()  # Khởi tạo model đúng kiến trúc
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

def preprocess(frame):
    # Chuyển BGR (OpenCV) sang RGB (PyTorch)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    ])
    return transform(frame).unsqueeze(0)

def predict(model, frame):
    input_tensor = preprocess(frame)
    with torch.no_grad():
        output = model(input_tensor)
        # Nếu output là tensor nhiều chiều, lấy giá trị đầu tiên
        if isinstance(output, (tuple, list)):
            output = output[0]
        # pred = torch.sigmoid(output).item()
        pred = torch.sigmoid(output).mean().item()
        print(f"Prediction score: {pred}")
        return "Real" if pred > 0.6 else "Fake"

def main():
    model_path = r"D:\E\DoANChuyenNganh\Facial_recognition\src\CDCNpp\CDCNpp_BinaryMask_P1_07_60.pkl"
    model = load_model(model_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không mở được camera!")
        return

    # label = "No face"
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # if detect_face(frame):  # Nếu có mặt
        if frame_count % 15 == 0:
            label = predict(model, frame)
        # else:
        #     label = "No face"  # Không phát hiện mặt

        frame_count += 1

        cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0) if label == "Real" else (0, 0, 255) if label == "Fake" else (0, 255, 255), 2)
        cv2.imshow('Face Anti-Spoofing', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()