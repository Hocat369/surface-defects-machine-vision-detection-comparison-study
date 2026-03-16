import cv2
import numpy as np
import time
import os
import glob

#----------------------------------------------
# Surface Defect Detection: Rule-based Approach
# Target Defect: Scratches
#----------------------------------------------

def rule_based_surface_defects_detection(folder_path):
    
    # 이미지 폴더 탐색
    image_paths = glob.glob(os.path.join(folder_path, '*.jpg'))

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
        
        # 1. Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Blurring (Gaussian Blur)
        # 3x3 픽셀 값의 가중 평균화
        blurred = cv2.GaussianBlur(gray, (3, 3), 0) 

        # 3. Sharpening
        # 중앙 픽셀을 강화, 주변 픽셀 약화로 대비 극대화
        # -1: 출력 이미지의 깊이(Depth)를 원본과 동일하게 유지
        sharpening_kernel = np.array([[-1, -1, -1],
                                    [-1,  9, -1],
                                    [-1, -1, -1]])
        sharpened = cv2.filter2D(blurred, -1, sharpening_kernel)

        # 4. Thresholding (Adaptive Thresholding)
        # .ADAPTIVE_THRESH_GAUSSIAN_C: 주변 픽셀의 가중 평균을 기준값으로 사용
        # .THRESH_BINARY_INV: 결함을 흰색(255), 배경을 검은색(0)으로 반전
        # 11: Block Size (11x11 구역 검사)
        # 2: 계산된 평균에서 인자만큼 제거해 최종 기준값을 여유 있게 설정
        thresh = cv2.adaptiveThreshold(
        sharpened, 255, 
        cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY, 
        31, -15
        )

        # 5. Morphology (Closing) 추
        # 작은 결함 점들을 하나의 덩어리로 연결
        kernel = np.ones((3, 3), np.uint8)
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        morph = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

        # 6. Contour & Bounding Box (Detection)
        # 결함 객체 검출
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result_img = img.copy()
        defect_count = 0

        img_area = img.shape[0] * img.shape[1] 
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # 면적 필터링
            # 10 픽셀 이상이면 결함으로 판정
            # 전체 면적의 70%가 넘어가는 거대한 덩어리는 무시
            if 10 < area < (img_area * 0.7): 
                defect_count += 1
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(result_img, f"NG", (x, y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 처리 완료 시간 및 FPS 계산
        end_time = time.time()
        p_time = (end_time - start_time) * 1000 
        processing_times.append(p_time)

        # 화면 출력
        cv2.imshow("0. Original", img)
        cv2.imshow("1. Grayscale", gray)
        cv2.imshow("2. Blurring", blurred)
        cv2.imshow("3. Sharpening", sharpened)
        cv2.imshow("4. Thresholding", thresh)
        cv2.imshow("5. Morphology", morph)
        cv2.imshow("6. Contour & Bounding Box ", result_img)

        # 파일명, 처리속도 출력
        file_name = os.path.basename(img_path)
        print(f"[{file_name}] 처리 완료 - {p_time:.2f} ms")

        # 5. 키보드 입력
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
    print("[Rule-based 결함 검출 성능 테스트 결과]")
    print(f" 총 테스트 이미지: {len(times_list)} 장")
    print(f" 평균 처리 시간: {avg_time:.2f} ms / 장")
    print(f" 평균 FPS: {avg_fps:.2f} FPS")
    print("=" * 30)



# 함수 실행 (폴더 경로)
rule_based_surface_defects_detection('01_OpenCV_Defects_Detection/NEU-DET/train/images/scratches')