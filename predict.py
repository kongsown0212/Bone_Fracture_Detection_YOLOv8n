import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.lite.python.interpreter import Interpreter  # hoặc `from tensorflow.lite.python.interpreter import Interpreter` nếu cần

# === 1. Load TFLite Model ===
interpreter = Interpreter(model_path=r"d:\Workspace\Deep_Learning\Object_detection\best_int8.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# === 2. Load và xử lý ảnh ===
image = cv2.imread(r"D:\Workspace\Deep_Learning\Object_detection\GRAZPEDWRI-DX-test\images\valid\0064_0245477750_01_WRI-L1_F009.png")  # ảnh cần test
image_input = cv2.resize(image, (640, 640))
input_tensor = image_input.astype(np.float32) / 255.0
input_tensor = np.expand_dims(input_tensor, axis=0)

# === 3. Tính toán FPS ===
start_time = time.time()  # Lấy thời gian bắt đầu

# === 4. Inference ===
interpreter.set_tensor(input_details[0]['index'], input_tensor)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])  # (1, 84, 8400)
print(output_data.shape)
output_data = np.squeeze(output_data).transpose(1, 0)  # -> (8400, 84)

# === 5. Hậu xử lý ===
boxes = []
conf_threshold = 0.3
for det in output_data:
    x, y, w_box, h_box = det[:4]
    class_scores = det[4:]
    class_id = np.argmax(class_scores)
    confidence = class_scores[class_id]

    if confidence > conf_threshold:
        x1 = int((x - w_box / 2) * image.shape[1])
        y1 = int((y - h_box / 2) * image.shape[0])
        x2 = int((x + w_box / 2) * image.shape[1])
        y2 = int((y + h_box / 2) * image.shape[0])
        boxes.append((x1, y1, x2, y2, confidence, class_id))

# === 6. Non-Max Suppression (đơn giản) ===
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1[:4]
    x1b, y1b, x2b, y2b = box2[:4]
    inter_x1 = max(x1, x1b)
    inter_y1 = max(y1, y1b)
    inter_x2 = min(x2, x2b)
    inter_y2 = min(y2, y2b)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2b - x1b) * (y2b - y1b)
    union = area1 + area2 - inter_area
    return inter_area / union if union != 0 else 0

def nms(boxes, iou_threshold=0.5):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    keep = []
    while boxes:
        box = boxes.pop(0)
        keep.append(box)
        boxes = [b for b in boxes if compute_iou(box, b) < iou_threshold]
    return keep

filtered_boxes = nms(boxes)  # Apply NMS to the filtered

# === 7. Vẽ kết quả ===
coco_classes = [ "boneanomaly","bonelesion","foreignbody","fracture"
,"metal"
,"periostealreaction"
,"pronatorsign"
,"softtissue"
,"text"
]

for x1, y1, x2, y2, conf, cls in filtered_boxes:
    class_name = coco_classes[int(cls)]
    label = f"{class_name}: {conf:.2f}"
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# === 8. Hiển thị bằng matplotlib ===
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12, 8))
plt.imshow(image_rgb)
plt.axis("off")
plt.title("YOLOv8m TFLite Detection")

# === 9. Tính toán FPS ===
end_time = time.time()  # Lấy thời gian kết thúc
print(end_time)
print(start_time)
fps = 1 / (end_time - start_time)  # Tính FPS
print(f"FPS: {fps:.2f}")

plt.show()