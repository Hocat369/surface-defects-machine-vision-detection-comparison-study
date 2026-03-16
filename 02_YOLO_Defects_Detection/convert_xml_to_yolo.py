import os
import xml.etree.ElementTree as ET
import glob

# 1. NEU-DET 데이터셋의 6가지 결함 종류 숫자 매핑
classes = {
    "crazing": 0,
    "inclusion": 1,
    "patches": 2,
    "pitted_surface": 3,
    "rolled-in_scale": 4,
    "scratches": 5
}

def convert_to_yolo(xml_dir, txt_dir):

    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)

    # 폴더 안의 모든 xml 파일 가져오기
    xml_files = glob.glob(os.path.join(xml_dir, '*.xml'))
    
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # 이미지 전체의 너비와 높이 가져오기
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        # txt 파일 이름 설정 (crazing_181.xml -> crazing_181.txt)
        txt_filename = os.path.basename(xml_file).replace('.xml', '.txt')
        txt_filepath = os.path.join(txt_dir, txt_filename)

        with open(txt_filepath, 'w') as out_file:

            # XML 안의 모든 <object> (결함 박스들) 반복
            for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                cls_name = obj.find('name').text
                
                if cls_name not in classes or int(difficult) == 1:
                    continue
                
                cls_id = classes[cls_name] # 'crazing'을 0으로 변환

                # bounding box 좌표 추출
                xmlbox = obj.find('bndbox')
                xmin = float(xmlbox.find('xmin').text)
                xmax = float(xmlbox.find('xmax').text)
                ymin = float(xmlbox.find('ymin').text)
                ymax = float(xmlbox.find('ymax').text)

                # YOLO 포맷으로 변환
                x_center = (xmin + xmax) / 2.0 / w
                y_center = (ymin + ymax) / 2.0 / h
                box_w = (xmax - xmin) / w
                box_h = (ymax - ymin) / h

                # 정규화된 값 txt 파일에 저장
                out_file.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")
        
    print(f"{len(xml_files)}개 XML 파일 TXT 포맷으로 변환 완료")


# XML, TXT 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
TRAIN_XML_FOLDER = os.path.join(current_dir, "NEU-DET_modified", "train", "annotations")
TRAIN_TXT_FOLDER = os.path.join(current_dir, "NEU-DET_modified", "train", "labels") 
VALIDATION_XML_FOLDER = os.path.join(current_dir, "NEU-DET_modified", "validation", "annotations")
VALIDATION_TXT_FOLDER = os.path.join(current_dir, "NEU-DET_modified", "validation", "labels") 

# 코드 실행
convert_to_yolo(TRAIN_XML_FOLDER, TRAIN_TXT_FOLDER)
convert_to_yolo(VALIDATION_XML_FOLDER, VALIDATION_TXT_FOLDER)