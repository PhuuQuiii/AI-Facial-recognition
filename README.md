Quí hướng dẫn:

Tải các package: pip install -r requirements.txt

run project: 
Nhận diện qua cam: 
python src/face_rec_cam.py 

Nhận diện qua video: python src/face_rec.py --path video/vdPhuQui.mp4 

Web:
cd : src
python face_rec_flask.py
Running on http://127.0.0.1:5000


------------------------------------
Tiền xử lý dữ liệu để cắt khuôn mặt từ ảnh gốc

python src/align_dataset_mtcnn.py  Dataset/FaceData/raw Dataset/FaceData/processed --image_size 160 --margin 32  --random_order --gpu_memory_fraction 0.25

Tiến hành train model để nhận diện khuôn mặt

python src/classifier.py TRAIN Dataset/FaceData/processed Models/20180402-114759.pb Models/facemodel.pkl --batch_size 1000


Tạo môi trường ảo tải package:
python -m venv venv
.\venv\Scripts\activate

pip install -r requirements.txt

