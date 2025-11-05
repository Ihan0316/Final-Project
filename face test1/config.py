# config.py
import os
import platform
from typing import List, Tuple, Dict


class Paths:
    """모델 및 데이터 경로 설정"""
    YOLO_VIOLATION_MODEL: str = "../model/best.pt"
    YOLO_POSE_MODEL: str = "../model/yolov8n-pose.pt"
    LOG_FOLDER: str = "violations"
    LOG_CSV: str = os.path.join(LOG_FOLDER, "system_log.csv")

    FAISS_INDEX: str = "face_index.faiss"
    FAISS_LABELS: str = "face_index.faiss.labels.npy"

    _os_name = platform.system()
    if _os_name == "Windows":
        FONT_PATH: str = "C:/Windows/Fonts/malgun.ttf"
    elif _os_name == "Darwin":
        FONT_PATH: str = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
    else:
        FONT_PATH: str = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

    EDSR_MODEL: str = "../model/EDSR_x4.pb"


class Thresholds:
    """탐지 및 인식 관련 임계값 설정"""
    SIMILARITY: float = 0.50
    YOLO_CONFIDENCE: float = 0.45
    POSE_CONFIDENCE: float = 0.50
    FALL_ASPECT_RATIO: float = 1.2
    FALL_TIME: float = 2.0
    FALL_MOVEMENT: int = 15
    MIN_FACE_SIZE: int = 32
    MIN_VISIBLE_KEYPOINTS: int = 6
    IOU_MATCHING: float = 0.60
    IOU_VIOLATION: float = 0.06
    MIN_VERTICAL_RATIO: float = 0.6
    UPSCALE_THRESHOLD: int = 100


class SystemConfig:
    """실시간 시스템 동작 관련 설정"""
    HEADLESS: bool = False
    CAMERA_INDICES: List[int] = [0]
    DISPLAY_WIDTH: int = 960
    DISPLAY_HEIGHT: int = 540
    MODEL_INPUT_WIDTH: int = 416
    MODEL_INPUT_HEIGHT: int = 416
    DETECTION_INTERVAL: int = 15
    MAX_INACTIVE_FRAMES: int = 100
    POSE_SMOOTHING_FACTOR: float = 0.4
    MAX_PEOPLE_TO_TRACK: int = 5
    EMBEDDING_BUFFER_SIZE: int = 5


class Policy:
    """이벤트 로그 및 재인식 정책 설정"""
    EVENT_LOG_COOLDOWN: int = 30
    UNKNOWN_RECOGNITION_COOLDOWN: int = 15
    KNOWN_RECOGNITION_COOLDOWN: int = 180


class Constants:
    """고정 상수 값"""
    SKELETON_CONNECTIONS: List[Tuple[int, int]] = [
        (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8),
        (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
    ]
    # --- [수정] 착용/미착용 클래스를 명확히 매핑하는 새로운 규칙 맵 ---
    SAFETY_RULES_MAP: Dict[str, Dict[str, str]] = {
        "안전모": {"compliance": "Hardhat", "violation": "NO-Hardhat"},
        "마스크": {"compliance": "Mask", "violation": "NO-Mask"},
        "안전조끼": {"compliance": "Safety Vest", "violation": "NO-Safety Vest"}
    }
    # --- [수정 완료] ---


os.makedirs(Paths.LOG_FOLDER, exist_ok=True)