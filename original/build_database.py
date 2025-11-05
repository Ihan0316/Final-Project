import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import time

# --- 설정 ---
DB_PATH = "image" # 얼굴 이미지 데이터베이스 경로
OUTPUT_FILE = "face_embeddings.npy" # 추출된 특징 데이터를 저장할 파일 이름
# ----------------

def build_database():
    """
    DB_PATH에 있는 모든 이미지로부터 얼굴 특징(임베딩)을 추출하여 파일로 저장합니다.
    """
    print("InsightFace 모델을 로딩합니다. 몇 초 정도 소요될 수 있습니다...")
    # 얼굴 분석을 위한 모델 로딩 (성능이 좋은 buffalo_l 모델 사용)
    # providers=['CUDAExecutionProvider']는 GPU를 사용하겠다는 의미입니다.
    # GPU가 없다면 providers=['CPUExecutionProvider']로 변경하세요.
    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    # DB의 모든 이미지를 처리하여 임베딩 추출
    face_database = {}
    
    start_time = time.time()
    image_count = 0

    # os.walk를 사용하여 하위 폴더까지 모두 탐색
    for root, dirs, files in os.walk(DB_PATH):
        for file in files:
            # 이미지 파일 확장자만 걸러내기
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                person_name = os.path.basename(root) # 폴더 이름을 사람 이름으로 사용
                
                print(f"처리 중: {image_path}")
                
                # OpenCV로 이미지 읽기
                img = cv2.imread(image_path)
                if img is None:
                    print(f"  [경고] 이미지를 읽을 수 없습니다: {image_path}")
                    continue
                
                # insightface 모델로 얼굴 탐지 및 특징 추출
                faces = app.get(img)
                
                if len(faces) == 0:
                    print(f"  [경고] 얼굴을 탐지할 수 없습니다: {image_path}")
                    continue
                
                # 가장 큰 얼굴 하나만 사용
                face = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)[0]
                embedding = face.normed_embedding # 512차원의 특징 벡터(임베딩)
                
                # face_database 딕셔너리에 저장
                if person_name not in face_database:
                    face_database[person_name] = []
                face_database[person_name].append(embedding)
                image_count += 1

    if not face_database:
        print("오류: DB 폴더에서 처리할 이미지를 찾을 수 없습니다. 경로를 확인하세요.")
        return

    # 추출된 데이터를 .npy 파일로 저장
    np.save(OUTPUT_FILE, face_database)
    
    end_time = time.time()
    print("-" * 30)
    print("✅ 얼굴 DB 구축 완료!")
    print(f"총 처리 시간: {end_time - start_time:.2f}초")
    print(f"총 인물 수: {len(face_database)}명")
    print(f"총 이미지 수: {image_count}개")
    print(f"저장된 파일: {OUTPUT_FILE}")
    print("-" * 30)

if __name__ == "__main__":
    build_database()