# utils.py
import logging
import datetime
import os
from typing import Tuple, List, Optional

import cv2
import faiss
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics.engine.results import Keypoints

import config


def setup_logging():
    """표준 로깅 시스템을 설정합니다."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.FileHandler("system.log"), logging.StreamHandler()]
    )
    logging.getLogger('PIL').setLevel(logging.WARNING)
    os.makedirs(config.Paths.LOG_FOLDER, exist_ok=True)


# --- [수정] OS에 맞는 폰트를 자동으로 로드하도록 개선 ---
try:
    if not os.path.exists(config.Paths.FONT_PATH):
        raise IOError(f"'{config.Paths.FONT_PATH}' 폰트 파일을 찾을 수 없습니다.")
    KOREAN_FONT = ImageFont.truetype(config.Paths.FONT_PATH, 14)
    logging.info(f"폰트 로드 성공: {config.Paths.FONT_PATH}")
except IOError as e:
    logging.warning(f"{e} 기본 폰트를 사용합니다. (한글이 깨질 수 있습니다)")
    KOREAN_FONT = ImageFont.load_default()
# --- [수정 완료] ---


class TextRenderer:
    """프레임의 모든 텍스트를 한 번에 그려 성능을 최적화하는 클래스."""

    def __init__(self, frame_shape: Tuple[int, int, int]):
        self.text_layer = Image.new("RGBA", (frame_shape[1], frame_shape[0]), (0, 0, 0, 0))
        self.draw = ImageDraw.Draw(self.text_layer)

    def add_text(self, text: str, pos: Tuple[int, int], bgr_color: Tuple[int, int, int]):
        x, y = pos
        rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
        text_bbox = self.draw.textbbox((x, y - 15), text, font=KOREAN_FONT)
        text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

        bg_y1 = y - text_h - 15
        bg_rect = (x - 2, bg_y1, x + text_w + 2, y)
        self.draw.rectangle(bg_rect, fill=(0, 0, 0, 128))
        self.draw.text((x, bg_y1), text, font=KOREAN_FONT, fill=(*rgb_color, 255))

    def render_on(self, frame: np.ndarray) -> np.ndarray:
        text_layer_rgba = np.array(self.text_layer)
        alpha_channel = text_layer_rgba[:, :, 3]
        y_coords, x_coords = np.where(alpha_channel > 0)
        if len(y_coords) == 0: return frame

        y_min, y_max = np.min(y_coords), np.max(y_coords)
        x_min, x_max = np.min(x_coords), np.max(x_coords)

        text_patch_rgba = text_layer_rgba[y_min:y_max + 1, x_min:x_max + 1]
        frame_patch = frame[y_min:y_max + 1, x_min:x_max + 1]

        alpha = (text_patch_rgba[:, :, 3] / 255.0)[:, :, np.newaxis]
        text_patch_bgr = text_patch_rgba[:, :, :3][:, :, ::-1]

        blended_patch = (frame_patch * (1 - alpha) + text_patch_bgr * alpha).astype(np.uint8)
        frame[y_min:y_max + 1, x_min:x_max + 1] = blended_patch
        return frame


def calculate_iou(box1: Tuple, box2: Tuple) -> float:
    x1_inter, y1_inter = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2_inter, y2_inter = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    if inter_area == 0: return 0.0
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def find_best_match_faiss(embedding: np.ndarray, faiss_index: faiss.Index,
                          labels: np.ndarray, threshold: float) -> Tuple[str, float]:
    """insightface 임베딩과 Faiss IndexFlatIP에 최적화된 검색 함수"""
    try:
        query_embedding = np.expand_dims(embedding.astype('float32'), axis=0)
        similarities, indices = faiss_index.search(query_embedding, 1)
        best_similarity = similarities[0][0]

        if best_similarity >= threshold:
            best_match_name = labels[indices[0][0]]
            return best_match_name, best_similarity
        return "Unknown", best_similarity
    except Exception as e:
        logging.error(f"Faiss 검색 중 오류 발생: {e}")
        return "Unknown", 0.0


def log_violation(frame: np.ndarray, person_name: str, event_type: str, cam_id: int):
    try:
        now = datetime.datetime.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        safe_event_type = "".join(c for c in event_type if c.isalnum() or c in ('-'))
        image_filename = f"{config.Paths.LOG_FOLDER}/{timestamp_str}_CAM{cam_id}_{person_name}_{safe_event_type}.jpg"
        cv2.imwrite(image_filename, frame)

        # --- [수정] config에서 CSV 파일 경로를 가져오도록 변경 ---
        log_filename = config.Paths.LOG_CSV
        log_entry = f"{now.strftime('%Y-%m-%d %H:%M:%S')},{person_name},{event_type},CAM-{cam_id},{image_filename}\n"
        if not os.path.exists(log_filename):
            with open(log_filename, 'w', encoding='utf-8-sig') as f:
                f.write("Timestamp,Person,Event,CameraID,EvidenceFile\n")
        with open(log_filename, 'a', encoding='utf-8-sig') as f:
            f.write(log_entry)
        logging.info(f"[CAM-{cam_id}] 이벤트 기록 저장: {person_name} - {event_type}")
    except Exception as e:
        logging.error(f"로그 파일/이미지 저장 실패: {e}")

# --- [삭제] 불필요한 스켈레톤 그리기 함수 제거 ---


def is_person_horizontal(keypoints: Keypoints, bbox_xyxy: Tuple) -> bool:
    try:
        points = keypoints.xy[0].cpu().numpy()
        confidences = keypoints.conf[0].cpu().numpy()

        valid_points = points[confidences > config.Thresholds.POSE_CONFIDENCE]
        if len(valid_points) < config.Thresholds.MIN_VISIBLE_KEYPOINTS:
            return False

        upper_indices = [0, 5, 6]
        lower_indices = [11, 12, 15, 16]

        valid_upper_y = [points[i][1] for i in upper_indices if confidences[i] > config.Thresholds.POSE_CONFIDENCE]
        valid_lower_y = [points[i][1] for i in lower_indices if confidences[i] > config.Thresholds.POSE_CONFIDENCE]

        if valid_upper_y and valid_lower_y:
            avg_upper_y = np.mean(valid_upper_y)
            avg_lower_y = np.mean(valid_lower_y)
            if avg_upper_y >= avg_lower_y:
                return True

        if not valid_lower_y: return False

        kpt_min_x, kpt_min_y = np.min(valid_points, axis=0)
        kpt_max_x, kpt_max_y = np.max(valid_points, axis=0)
        kpt_width, kpt_height = kpt_max_x - kpt_min_x, kpt_max_y - kpt_min_y

        if kpt_height < 1e-5: return True

        kpt_aspect_ratio = kpt_width / kpt_height
        return kpt_aspect_ratio > config.Thresholds.FALL_ASPECT_RATIO

    except Exception as e:
        logging.warning(f"넘어짐 감지 함수(is_person_horizontal) 오류: {e}")
        return False


def get_keypoints_center(keypoints: Keypoints) -> Optional[np.ndarray]:
    try:
        points = keypoints.xy[0].cpu().numpy()
        confidences = keypoints.conf[0].cpu().numpy()
        valid_points = points[confidences > config.Thresholds.POSE_CONFIDENCE]
        if len(valid_points) > 0: return np.mean(valid_points, axis=0)
    except Exception:
        return None


def clip_bbox_xyxy(bbox_xyxy: Tuple, frame_w: int, frame_h: int) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_w, x2), min(frame_h, y2)
    if (x2 - x1) > 0 and (y2 - y1) > 0:
        return x1, y1, x2, y2
    return None