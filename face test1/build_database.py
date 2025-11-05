import os
import sys
import platform
import cv2
import numpy as np
import onnxruntime
from insightface.app import FaceAnalysis
import time
import faiss  # Faiss 임포트
import config  # config.py에서 경로를 가져오기 위해 임포트

# --- 설정 ---
DB_PATH = "../image"  # 얼굴 이미지 데이터베이스 경로
# config.py와 경로를 맞춥니다.
OUTPUT_EMBEDDINGS = "face_embeddings.npy"  # 백업용 원본
FAISS_INDEX_FILE = config.Paths.FAISS_INDEX
FAISS_LABELS_FILE = config.Paths.FAISS_LABELS


# ----------------

def print_runtime_environment():
    """
    현재 실행 환경 (OS, Python, 라이브러리 버전)을 출력합니다.
    """
    print("-" * 30)
    print("🚀 실행 환경 확인 🚀")
    print(f"  - 운영체제: {platform.system()} {platform.release()}")
    print(f"  - Python 버전: {sys.version}")
    print(f"  - ONNX Runtime 버전: {onnxruntime.__version__}")
    # 사용 가능한 실행 프로바이더 목록을 출력하여 MPS/GPU 가속 가능 여부를 확인합니다.
    print(f"  - ONNX 사용 가능 Providers: {onnxruntime.get_available_providers()}")
    print(f"  - OpenCV 버전: {cv2.__version__}")
    print(f"  - NumPy 버전: {np.__version__}")
    print("-" * 30)


def build_database():
    """
    DB_PATH에 있는 모든 이미지로부터 얼굴 특징(임베딩)을 추출하여 Faiss 인덱스를 생성합니다.
    각 이미지에 대해 원본, 좌우 반전, 밝기 조절 등 데이터 증강을 적용합니다.
    """
    print("InsightFace 모델을 로딩합니다. 몇 초 정도 소요될 수 있습니다...")
    # Apple Silicon GPU(MPS)를 사용하기 위해 CoreMLExecutionProvider로 변경합니다.
    providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640))

    # 현재 사용 중인 실행 제공자(Provider) 확인 및 출력
    print(f"✅ 모델 로딩 완료. 사용 중인 Provider: {app.models['detection'].session.get_providers()}")

    # DB의 모든 이미지를 처리하여 임베딩 추출
    face_database = {}  # 임시 저장용
    start_time = time.time()
    processed_files_count = 0
    embedding_count = 0

    # os.walk를 사용하여 하위 폴더까지 모두 탐색
    for root, dirs, files in os.walk(DB_PATH):
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            continue

        for file in image_files:
            image_path = os.path.join(root, file)
            person_name = os.path.basename(root)  # 폴더 이름을 사람 이름으로 사용

            print(f"처리 중: {image_path} (원본 + 증강 3종)")

            # OpenCV로 이미지 읽기
            img = cv2.imread(image_path)
            if img is None:
                print(f"  [경고] 이미지를 읽을 수 없습니다: {image_path}")
                continue

            processed_files_count += 1

            # --- [수정] 데이터 증강 적용 ---
            # 처리할 이미지들을 리스트에 담습니다.
            images_to_process = []

            # 1. 원본 이미지
            images_to_process.append(img)
            # 2. 좌우 반전 이미지
            images_to_process.append(cv2.flip(img, 1))
            # 3. 밝기 증가 이미지 (alpha: 대비, beta: 밝기)
            images_to_process.append(cv2.convertScaleAbs(img, alpha=1.0, beta=40))
            # 4. 밝기 감소 이미지
            images_to_process.append(cv2.convertScaleAbs(img, alpha=1.0, beta=-40))
            # --- [수정 완료] ---

            # 원본 및 증강된 모든 이미지에서 특징 추출
            for augmented_img in images_to_process:
                faces = app.get(augmented_img)

                if not faces:
                    # 얼굴을 찾지 못한 경우, 경고 메시지 없이 그냥 넘어갑니다.
                    # (예: 좌우 반전 시 얼굴이 아닐 수 있음)
                    continue

                # 가장 큰 얼굴 하나만 사용
                face = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)[0]
                embedding = face.normed_embedding  # 512차원의 특징 벡터(임베딩)

                # face_database 딕셔너리에 저장
                if person_name not in face_database:
                    face_database[person_name] = []
                face_database[person_name].append(embedding)
                embedding_count += 1

    if not face_database:
        print("오류: DB 폴더에서 처리할 이미지를 찾을 수 없습니다. 경로를 확인하세요.")
        return

    # Faiss 인덱스 구축 로직
    print("DB 구축 완료. Faiss 인덱스를 생성합니다...")
    labels_list = []
    embeddings_list = []

    # 딕셔너리를 Faiss가 사용할 리스트로 변환
    for name, embeddings in face_database.items():
        for embedding in embeddings:
            labels_list.append(name)
            embeddings_list.append(embedding)

    if not embeddings_list:
        print("오류: 추출된 임베딩이 없습니다.")
        return

    embeddings_array = np.array(embeddings_list).astype('float32')
    labels_array = np.array(labels_list)
    d = embeddings_array.shape[1]  # 임베딩 차원 (512)

    # 내적(IP)은 코사인 유사도와 동일합니다. IndexFlatIP를 사용합니다.
    index = faiss.IndexFlatIP(d)
    index.add(embeddings_array)

    # Faiss 인덱스와 라벨 배열 저장
    faiss.write_index(index, FAISS_INDEX_FILE)
    np.save(FAISS_LABELS_FILE, labels_array)

    # 기존 .npy 저장 (백업용)
    np.save(OUTPUT_EMBEDDINGS, face_database)

    end_time = time.time()
    print("-" * 30)
    print("✅ Faiss 인덱스 및 데이터베이스 구축 완료!")
    print(f"총 처리 시간: {end_time - start_time:.2f}초")
    print(f"처리한 원본 이미지 수: {processed_files_count}개")
    print(f"총 인물 수: {len(face_database)}명")
    print(f"총 임베딩 수 (증강 포함): {len(labels_list)}개")
    print(f"저장된 인덱스: {FAISS_INDEX_FILE}")
    print(f"저장된 라벨: {FAISS_LABELS_FILE}")
    print(f"(참고용) 원본 DB: {OUTPUT_EMBEDDINGS}")
    print("-" * 30)


if __name__ == "__main__":
    print_runtime_environment()
    build_database()