# Import YOLO class
from ultralytics import YOLO
u
# =========Load mô hình pretrained hoặc resume từ checkpoint =========
resume_training = True  # ← Đặt True nếu muốn tiếp tục từ last.pt

if resume_training:
    model = YOLO(r"D:\Workspace\Deep_Learning\Object_detection\runs\detect_light\weights\last.pt")  # load lại model cũ
else:
    model = YOLO('yolov8n.pt')  # khởi tạo model YOLOv8n với pretrained weights

# =========Huấn luyện =========
model.train(
    data=r"d:/Workspace/Deep_Learning/Object_detection/GRAZPEDWRI-DX/data/meta.yaml",
    epochs=70,          
    batch=32,
    imgsz=640,
    pretrained=False,   
    freeze=10,          
    optimizer='SGD',
    lr0=0.005,          
    device='cpu',
    augment=True,
    project='runs',
    name='detect_light',
    exist_ok=True,
    resume=True,     
    workers=2
)

# ========= Đánh giá mô hình sau huấn luyện =========
metrics = model.val()       
          

