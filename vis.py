# 调用模型可视化

from ultralytics import YOLO
import cv2
import os

model = YOLO("runs/detect/train8/weights/best.pt")

results = model.predict(
    source="./sz_data332_x/images/val",
    save=False,       # 禁止自动保存
    verbose=False     # 可选：关闭详细输出
)

save_dir = "runs/detect/predict332"
os.makedirs(save_dir, exist_ok=True)

for r in results:
    im_array = r.plot(
        line_width=1,  # 细边框
        labels=False,  # 不显示标签
        conf=False     # 不显示置信度
    )
    save_path = os.path.join(save_dir, r.path.split("/")[-1])
    cv2.imwrite(save_path, im_array)

print(f"结果已保存到 {save_dir}")