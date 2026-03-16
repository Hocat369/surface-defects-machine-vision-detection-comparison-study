from ultralytics import YOLO, settings
import os

if __name__ == '__main__':

    # Set relative paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    # 1. pre-trained 모델 불러오기
    model = YOLO('yolov8n.pt')    

    # 2. Train 설정 
    results = model.train(
        data='data.yaml',      
        epochs=100,             
        imgsz=200,             
        batch=16,
        name='result_model'  
    )

    print("학습 완료")