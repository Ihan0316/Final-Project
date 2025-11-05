# config.py - 시스템 설정 및 구성
import os
import logging
import platform
from typing import List, Tuple, Dict
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 이 파일의 실제 위치(src 폴더)를 기준으로 절대 경로를 생성합니다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # -> /app/src

class Paths:
    """모델 및 데이터 경로 설정"""
    # 환경변수에서 모델 경로를 가져오거나 기본값 사용

    # 모델 경로 설정 (/app/src에서 한 단계 위인 /app 폴더 아래의 model 폴더)
    MODEL_BASE_DIR = os.getenv('MODEL_BASE_DIR', os.path.join(BASE_DIR, "../model"))

    YOLO_VIOLATION_MODEL: str = os.path.normpath(os.path.join(MODEL_BASE_DIR, "best.pt"))
    YOLO_POSE_MODEL: str = os.path.normpath(os.path.join(MODEL_BASE_DIR, "yolov8n-pose.pt"))
    EDSR_MODEL: str = os.path.normpath(os.path.join(MODEL_BASE_DIR, "EDSR_x4.pb"))

    LOG_FOLDER: str = os.path.join(BASE_DIR, "violations") # -> /app/src/violations
    LOG_CSV: str = os.path.join(LOG_FOLDER, "system_log.csv")

    # FAISS 파일은 /app 루트에 복사됨
    FAISS_INDEX: str = os.path.normpath(os.path.join(BASE_DIR, "../face_index.faiss"))
    FAISS_LABELS: str = os.path.normpath(os.path.join(BASE_DIR, "../face_index.faiss.labels.npy"))

    _os_name = platform.system()
    if _os_name == "Windows":
        FONT_PATH: str = "C:/Windows/Fonts/malgun.ttf"
    elif _os_name == "Darwin":
        FONT_PATH: str = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
    else:
        FONT_PATH: str = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

class Thresholds:
    """탐지 및 인식 관련 임계값 설정 - 정확도 향상"""
    SIMILARITY: float = float(os.getenv('SIMILARITY_THRESHOLD', '0.5'))  # 유사도 임계값 조정 (실용적 수준)

    # 객체 탐지 임계값 (감지율 향상을 위해 임계값 최적화)
    YOLO_CONFIDENCE: float = float(os.getenv('YOLO_CONFIDENCE', '0.25'))  # 감지율 향상을 위해 임계값 하향
    POSE_CONFIDENCE: float = float(os.getenv('POSE_CONFIDENCE', '0.20'))  # 포즈 감지율 향상을 위해 임계값 하향

    # 추적 관련 임계값 (정확도 향상)
    IOU_MATCHING: float = float(os.getenv('IOU_MATCHING', '0.6'))  # 매칭 IoU 상향으로 더 정확한 매칭
    IOU_VIOLATION: float = float(os.getenv('IOU_VIOLATION', '0.15'))  # 위반 탐지 IoU 상향으로 정확도 향상

    # 넘어짐 감지 (개선된 로직)
    FALL_ASPECT_RATIO: float = 1.2
    FALL_TIME: float = float(os.getenv('FALL_TIME', '1.5'))  # 넘어짐 감지 시간 단축
    FALL_MOVEMENT: int = 15

    # 넘어짐 감지 세부 임계값 (개선된 로직용)
    FALL_VERTICAL_RATIO_THRESHOLD: float = 0.3  # 수직 비율 임계값 (낮을수록 넘어짐 가능성 높음)
    FALL_HORIZONTAL_SPREAD_RATIO: float = 1.5  # 수평 분산 비율 임계값
    FALL_SCORE_THRESHOLD: float = 0.6  # 넘어짐 점수 임계값 (0.6 이상이면 넘어짐으로 판단)

    # 얼굴 인식 중복 제거 설정 (정확도 향상)
    FACE_IOU_THRESHOLD: float = 0.4  # 얼굴 중복 제거 IoU 임계값 상향 (더 엄격한 중복 제거)
    MIN_FACE_SIZE_DEDUP: int = 40  # 중복 제거 시 최소 얼굴 크기 증가 (더 큰 얼굴만 처리)

    # 얼굴 인식 (인식률 향상)
    MIN_FACE_SIZE: int = 30  # 최소 얼굴 크기 감소로 인식률 향상
    UPSCALE_THRESHOLD: int = 100  # 업스케일 임계값 감소로 인식률 향상

    # 포즈 감지 품질 (정확도 향상)
    MIN_VISIBLE_KEYPOINTS: int = 10  # 최소 키포인트 수 증가로 정확도 향상
    MIN_VERTICAL_RATIO: float = 0.7  # 수직 비율 임계값 상향으로 정확도 향상

    # 사람 탐지 필터링 (손/작은 객체 오인식 방지) - 조금 더 엄격하게 상향
    MIN_PERSON_BOX_WIDTH: int = 60   # 40 -> 60
    MIN_PERSON_BOX_HEIGHT: int = 120 # 80 -> 120
    MIN_PERSON_BOX_AREA: int = 7200  # 3200 -> 7200 (60*120)
    MAX_PERSON_ASPECT_RATIO: float = 2.0  # 2.5 -> 2.0
    MIN_PERSON_ASPECT_RATIO: float = 0.5  # 0.3 -> 0.5

    # 상반신만 보이는 경우를 위한 완화 임계값
    RELAXED_MIN_PERSON_BOX_WIDTH: int = 40
    RELAXED_MIN_PERSON_BOX_HEIGHT: int = 80
    RELAXED_MIN_PERSON_BOX_AREA: int = 3200
    RELAXED_MAX_PERSON_ASPECT_RATIO: float = 2.5
    RELAXED_MIN_PERSON_ASPECT_RATIO: float = 0.3


class SystemConfig:
    """실시간 시스템 동작 관련 설정 - MPS 최적화"""
    HEADLESS: bool = False

    @classmethod
    def get_device_config(cls) -> Dict[str, any]:
        """디바이스 설정 (CUDA 우선)"""
        import torch
        import platform

        device_config = {'device': 'cpu'}

        if torch.cuda.is_available():
            device_config['device'] = 'cuda'
            logging.info("CUDA GPU 활성화됨")
        elif platform.system() == 'Darwin' and torch.backends.mps.is_available():
            device_config['device'] = 'mps'
            logging.info("MPS (Metal Performance Shaders) 활성화됨")
        else:
            logging.warning("GPU를 사용할 수 없습니다. CPU 모드로 실행됩니다.")

        return device_config

    @classmethod
    def get_camera_indices(cls) -> List[int]:
        """환경 변수에서 카메라 인덱스를 가져옵니다."""
        indices_str = os.getenv('CAMERA_INDICES', '0')  # 기본값을 0으로 변경

        if indices_str.lower() == 'auto':
            # 서버 환경에서는 자동 감지가 의미 없을 수 있음
            logging.warning("CAMERA_INDICES=auto. 서버 환경에서는 지원되지 않을 수 있습니다. [0] 사용")
            return [0]

        try:
            cameras = [int(x.strip()) for x in indices_str.split(',')]
            logging.info(f"환경 변수에서 설정된 카메라: {cameras}")
            return cameras
        except ValueError:
            logging.warning("CAMERA_INDICES 환경 변수 형식 오류. 기본값 [0] 사용")
            return [0]

    # (server.py가 카메라를 직접 제어하지 않으므로 이 설정들은 대부분 client.py로 이동)
    CAMERA_INDICES: List[int] = [0]

    # 디스플레이 해상도 (서버 처리 기준)
    DISPLAY_WIDTH: int = int(os.getenv('DISPLAY_WIDTH', '960'))
    DISPLAY_HEIGHT: int = int(os.getenv('DISPLAY_HEIGHT', '540'))

    # 모델 입력 해상도 - Tesla V100 최적화
    MODEL_INPUT_WIDTH: int = int(os.getenv('MODEL_INPUT_WIDTH', '320'))  # 256 → 320 (정확도 향상)
    MODEL_INPUT_HEIGHT: int = int(os.getenv('MODEL_INPUT_HEIGHT', '320'))  # 256 → 320

    DETECTION_INTERVAL: int = int(os.getenv('DETECTION_INTERVAL', '1'))
    MAX_INACTIVE_FRAMES: int = int(os.getenv('MAX_INACTIVE_FRAMES', '60'))
    POSE_SMOOTHING_FACTOR: float = 0.4
    MAX_PEOPLE_TO_TRACK: int = int(os.getenv('MAX_PEOPLE_TO_TRACK', '10'))
    EMBEDDING_BUFFER_SIZE: int = int(os.getenv('EMBEDDING_BUFFER_SIZE', '5'))

    # 성능 최적화 (CUDA 기준) - Tesla V100 최적화
    ENABLE_HALF_PRECISION: bool = True
    ENABLE_MODEL_FUSION: bool = True
    ENABLE_TENSORRT: bool = True  # CUDA 환경에서 TensorRT 사용 고려 (YOLO 모델 최적화)
    ENABLE_ONNX_OPTIMIZATION: bool = True
    ENABLE_MODEL_QUANTIZATION: bool = True
    BATCH_SIZE: int = int(os.getenv('BATCH_SIZE', '8'))  # 4 → 8 (Tesla V100 최적화)
    MAX_WORKERS: int = int(os.getenv('MAX_WORKERS', '8'))  # 4 → 8 (더 많은 병렬 처리)

    ENABLE_MODEL_CACHING: bool = True
    ENABLE_FRAME_SKIPPING: bool = True
    FRAME_SKIP_RATIO: float = 0.3
    ENABLE_ADAPTIVE_QUALITY: bool = True


class Policy:
    """이벤트 로그 및 재인식 정책 설정"""
    EVENT_LOG_COOLDOWN: int = int(os.getenv('EVENT_LOG_COOLDOWN', '15'))
    UNKNOWN_RECOGNITION_COOLDOWN: int = int(os.getenv('UNKNOWN_RECOGNITION_COOLDOWN', '10'))
    KNOWN_RECOGNITION_COOLDOWN: int = int(os.getenv('KNOWN_RECOGNITION_COOLDOWN', '120'))


class Constants:
    """고정 상수 값"""
    SKELETON_CONNECTIONS: List[Tuple[int, int]] = [
        (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8),
        (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
    ]
    SAFETY_RULES_MAP: Dict[str, Dict[str, str]] = {
        "안전모": {"compliance": "Hardhat", "violation": "NO-Hardhat"},
        "마스크": {"compliance": "Mask", "violation": "NO-Mask"},
        "안전조끼": {"compliance": "Safety Vest", "violation": "NO-Safety Vest"}
        # 장갑(Glove)은 사용자 요청으로 제외
    }

# 로그 폴더 생성
os.makedirs(Paths.LOG_FOLDER, exist_ok=True)
