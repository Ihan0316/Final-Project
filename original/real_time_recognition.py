import cv2
import numpy as np
from insightface.app import FaceAnalysis
from numpy.linalg import norm
import time

# --- 설정 ---
EMBEDDINGS_FILE = "face_embeddings.npy" # 1단계에서 생성한 DB 파일
# 이 값보다 유사도가 높아야 동일인으로 판단합니다. (0.5가 적당한 시작점)
SIMILARITY_THRESHOLD = 0.4
# ----------------

def load_database(file_path):
    """
    저장된 얼굴 특징 데이터(.npy)를 불러옵니다.
    """
    try:
        # allow_pickle=True는 딕셔너리 형태를 불러오기 위해 필요합니다.
        database = np.load(file_path, allow_pickle=True).item()
        print("✅ 얼굴 DB 로딩 성공!")
        print(f"   등록된 인물: {list(database.keys())}")
        return database
    except FileNotFoundError:
        print(f"오류: '{file_path}' 파일을 찾을 수 없습니다.")
        print("먼저 build_database.py를 실행하여 DB를 생성해주세요.")
        return None


def find_best_match(embedding, database, threshold):
    """
    입력된 얼굴 특징(embedding)과 DB에 있는 모든 얼굴을 비교하여 가장 일치하는 사람을 찾습니다.
    """
    best_match_name = "Unknown"
    max_similarity = -1 # 유사도는 -1 ~ 1 사이의 값을 가집니다.

    for name, embeddings in database.items():
        for db_embedding in embeddings:
            # 코사인 유사도 계산: 두 벡터가 얼마나 같은 방향을 가리키는지를 측정
            # A·B / (||A|| * ||B||)
            similarity = np.dot(embedding, db_embedding) / (norm(embedding) * norm(db_embedding))
            
            if similarity > max_similarity:
                max_similarity = similarity
                # 임계값을 넘어야만 이름을 할당
                if similarity > threshold:
                    best_match_name = name

    return best_match_name, max_similarity


def main():
    # 1. 얼굴 DB 로드
    face_database = load_database(EMBEDDINGS_FILE)
    if face_database is None:
        return

    # 2. InsightFace 모델 로드 (DB 생성 때와 동일하게 설정)
    print("InsightFace 모델을 로딩합니다...")
    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    # 3. 웹캠 열기
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("오류: 웹캠을 열 수 없습니다.")
        return
        
    print("\n실시간 얼굴 인식을 시작합니다. 종료하려면 'q'를 누르세요.")
    p_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 4. 실시간 영상에서 얼굴 탐지
        faces = app.get(frame)
        
        # 탐지된 각 얼굴에 대해 처리
        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            
            # 5. 얼굴 특징(임베딩) 추출
            embedding = face.normed_embedding

            # 6. DB와 비교하여 가장 일치하는 사람 찾기
            name, similarity = find_best_match(embedding, face_database, SIMILARITY_THRESHOLD)

            # 7. 결과 표시
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 유사도를 소수점 둘째 자리까지 표시
            text = f"{name} ({similarity:.2f})"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # FPS 계산 및 표시
        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time
        cv2.putText(frame, f"FPS: {int(fps)}", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Real-time Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("프로그램을 종료합니다.")

if __name__ == "__main__":
    main()