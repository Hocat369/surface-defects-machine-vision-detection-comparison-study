import cv2
import time
import os
import glob
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

#----------------------------------------------------------
# Surface Defect Detection: Machine Learning-based Apporach
# AI Model: YOLOv8n
# Target Defect: Scratches
#----------------------------------------------------------

def yolo_surface_defects_detection(folder_path, model_path):
    
    # 1. YOLO 모델 로드
    model = YOLO(model_path)
    
    # 2. 이미지 폴더 탐색
    image_paths = glob.glob(os.path.join(folder_path, '*.*'))

    if not image_paths:
        print(f"'{folder_path}' 폴더에서 이미지를 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    print(f"총 이미지 갯수: {len(image_paths)}장\n")
    print("단축키 안내: [n] 다음 사진으로 넘어가기 / [q] 테스트 종료")
    print("-" * 50)

    # 평균 처리 시간 측정용 리스트
    processing_times = []

    for img_path in image_paths:

        # 0. 이미지 로드
        img = cv2.imread(img_path)
        if img is None:
            print(f"이미지 로드 실패 (스킵): {img_path}")
            continue

        # 영상처리 속도(FPS) 측정 시작
        start_time = time.time()
        
        # 1. YOLO 추론 (이미 읽어온 img 배열을 그대로 전달해 속도 최적화)
        # verbose=False: 터미널 출력 off
        results = model(img, verbose=False) 
        
        # 2. 결과 이미지 생성
        annotator = Annotator(img.copy(), line_width=2, font_size=4)
        
        for box in results[0].boxes:
            b = box.xyxy[0]                
            cls = int(box.cls)             
            label_name = model.names[cls]  
            conf = box.conf[0]             

            # 박스 및 텍스트 색상 설정
            my_box_color = (0, 255, 0)     
            my_text_color = (0, 0, 0)  

            annotator.box_label(b, 
                                label=f"{label_name} {conf:.2f}",
                                color=my_box_color,   
                                txt_color=my_text_color 
                                )

        # 그려진 이미지 반환
        result_img = annotator.result()

        # 처리 완료 시간 및 FPS 계산
        end_time = time.time()
        p_time = (end_time - start_time) * 1000 
        processing_times.append(p_time)

        # 화면 출력
        #resized_img = cv2.resize(result_img, (0, 0), fx=2, fy=2)
        cv2.imshow("Origianl", img)
        cv2.imshow("YOLO Detection Result", result_img)

        # 파일명, 처리속도 출력
        file_name = os.path.basename(img_path)
        print(f"[{file_name}] 처리 완료 - {p_time:.2f} ms")

        # 4. 키보드 입력
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('n'):
                break 
            elif key == ord('q'):
                print("\n사용자에 의해 검사가 중단되었습니다.")
                cv2.destroyAllWindows()
                
                # 중단 시점까지의 평균 시간 출력 후 함수 완전 종료
                print_average_stats(processing_times)
                return 
    
    # 모든 사진 검사 완료
    cv2.destroyAllWindows()
    print("\n표면 결함 검출 테스트 완료")
    print_average_stats(processing_times)

# 평균 통계 함수 
def print_average_stats(times_list):
    if not times_list:
        return
    
    avg_time = sum(times_list) / len(times_list)
    avg_fps = 1000 / avg_time if avg_time > 0 else 0
    
    print("=" * 30)
    print("[YOLO 결함 검출 성능 테스트 결과]")
    print(f" 총 테스트 이미지: {len(times_list)} 장")
    print(f" 평균 처리 시간: {avg_time:.2f} ms / 장")
    print(f" 평균 FPS: {avg_fps:.2f} FPS")
    print("=" * 30)

if __name__ == '__main__':
    # 테스트할 폴더 경로, 학습된 모델 경로 지정
    test_images_folder = '02_YOLO_Defects_Detection/NEU-DET_modified/validation/images'
    trained_model_path = '02_YOLO_Defects_Detection/runs/detect/result_model/weights/best.pt'
    
    # 함수 실행
    yolo_surface_defects_detection(test_images_folder, trained_model_path)