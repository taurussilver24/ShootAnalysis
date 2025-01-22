from ultralytics import YOLO
import torch


if __name__ == "__main__":

    # モデルとデバイス選択
    model = YOLO('Yolo-Weights/yolov8n.pt')
    print(torch.cuda.is_available())

    # モデル学習
    results = model.train(data='cvat.yaml', epochs=100, imgsz=640, batch=64, device=0, amp=True)

