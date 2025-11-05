# utils.py (최종 수정본)
import datetime
import logging
import os
from typing import Tuple, Optional, List
import cv2
import faiss
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics.engine.results import Keypoints

import config


def setup_logging():
    """표준 로깅 시스템을 설정합니다."""
    import logging.handlers

    # 로그 폴더 생성
    os.makedirs(config.Paths.LOG_FOLDER, exist_ok=True)

    # 로그 파일 핸들러 설정 (로테이션 포함)
    file_handler = logging.handlers.RotatingFileHandler(
        # ⭐️ 로그 파일 경로를 config에서 가져오도록 수정 ⭐️
        os.path.join(config.Paths.LOG_FOLDER, "system.log"),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )

    # 콘솔 핸들러 설정
    console_handler = logging.StreamHandler()

    # 포맷터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s', # ⭐️ 포맷터에 파일명/줄번호 추가 ⭐️
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 로거 설정
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler]
    )

    # PIL 로거 레벨 조정
    logging.getLogger('PIL').setLevel(logging.WARNING)

    logging.info("로깅 시스템 초기화 완료")


def create_standard_response(data=None, status="success", message="", error_code=None):
    """표준화된 API 응답 형식을 생성합니다."""
    response = {
        "status": status,
        "message": message,
        "timestamp": datetime.datetime.now().isoformat(),
        "data": data
    }
    if error_code:
        response["error_code"] = error_code
    return response


# --- [수정] OS에 맞는 폰트를 자동으로 로드하도록 개선 ---
try:
    if not os.path.exists(config.Paths.FONT_PATH):
        # ⭐️ Docker 이미지 내의 대체 폰트 경로 시도 ⭐️
        fallback_font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        if os.path.exists(fallback_font):
             logging.warning(f"Nanum 폰트({config.Paths.FONT_PATH}) 없음. {fallback_font} 사용.")
             config.Paths.FONT_PATH = fallback_font
        else:
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

        try:
            # 텍스트 크기 계산
            text_bbox = self.draw.textbbox((0, 0), text, font=KOREAN_FONT)
            text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        except Exception as e:
            logging.warning(f"텍스트 렌더링 크기 계산 오류: {e} (텍스트: {text})")
            text_w, text_h = 50, 10 # 기본 크기

        # 화면 경계 정보 가져오기
        frame_h, frame_w = self.text_layer.size[1], self.text_layer.size[0]

        # 텍스트 위치를 화면 경계 내로 제한
        # x 좌표 제한 (텍스트가 오른쪽 경계를 넘지 않도록)
        if x + text_w + 2 > frame_w:
            x = frame_w - text_w - 2
        if x < 2:
            x = 2

        # y 좌표 제한 (텍스트가 위쪽 경계를 넘지 않도록)
        bg_y1 = y - text_h - 5 # ⭐️ 패딩 조정 ⭐️
        if bg_y1 < 0:
            bg_y1 = 0
            y = bg_y1 + text_h + 5

        # 배경 사각형 그리기 (경계 내에서)
        bg_rect = (x - 2, bg_y1, x + text_w + 2, y)
        self.draw.rectangle(bg_rect, fill=(0, 0, 0, 128))

        # 텍스트 그리기
        try:
            self.draw.text((x, bg_y1), text, font=KOREAN_FONT, fill=(*rgb_color, 255))
        except Exception as e:
            logging.warning(f"텍스트 렌더링 그리기 오류: {e}")


    def render_on(self, frame: np.ndarray) -> np.ndarray:
        try:
            text_layer_rgba = np.array(self.text_layer)
            alpha_channel = text_layer_rgba[:, :, 3]

            # 알파 채널에 내용이 있는지 확인
            if not np.any(alpha_channel > 0):
                return frame

            y_coords, x_coords = np.where(alpha_channel > 0)

            # 좌표가 비어있는 극단적인 경우 방지
            if len(y_coords) == 0 or len(x_coords) == 0:
                return frame

            y_min, y_max = np.min(y_coords), np.max(y_coords)
            x_min, x_max = np.min(x_coords), np.max(x_coords)

            # 프레임 경계를 넘지 않도록 보정
            y_max = min(y_max, frame.shape[0] - 1)
            x_max = min(x_max, frame.shape[1] - 1)
            y_min = max(0, y_min)
            x_min = max(0, x_min)

            text_patch_rgba = text_layer_rgba[y_min:y_max + 1, x_min:x_max + 1]
            frame_patch = frame[y_min:y_max + 1, x_min:x_max + 1]

            # 크기 일치 확인
            if text_patch_rgba.shape[:2] != frame_patch.shape[:2]:
                 logging.warning(f"TextRenderer: 패치 크기 불일치! Text={text_patch_rgba.shape}, Frame={frame_patch.shape}. 렌더링 건너뜀.")
                 return frame

            alpha = (text_patch_rgba[:, :, 3] / 255.0)[:, :, np.newaxis]
            text_patch_bgr = text_patch_rgba[:, :, :3][:, :, ::-1] # RGBA -> BGR

            blended_patch = (frame_patch * (1 - alpha) + text_patch_bgr * alpha).astype(np.uint8)
            frame[y_min:y_max + 1, x_min:x_max + 1] = blended_patch
            return frame
        except Exception as e:
            logging.error(f"텍스트 렌더링 적용(render_on) 오류: {e}", exc_info=True)
            return frame


def calculate_iou(box1: Tuple, box2: Tuple) -> float:
    try:
        x1_inter, y1_inter = max(box1[0], box2[0]), max(box1[1], box2[1])
        x2_inter, y2_inter = min(box1[2], box2[2]), min(box1[3], box2[3])
        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        if inter_area == 0: return 0.0
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0
    except Exception as e:
        logging.warning(f"IOU 계산 오류: {e} (box1={box1}, box2={box2})")
        return 0.0


def find_best_match_faiss(embedding: np.ndarray, faiss_index: faiss.Index,
                          threshold: float) -> Tuple[str, float]:
    """insightface 임베딩과 Faiss IndexFlatIP에 최적화된 검색 함수"""
    try:
        # ⭐️ 레이블 파일(.npy)을 여기서 로드 (config 경로 사용) ⭐️
        if not hasattr(find_best_match_faiss, "labels"):
             label_path = config.Paths.FAISS_LABELS
             # ⭐️ Dockerfile WORKDIR 변경으로 경로 수정: /app/face_index.faiss.labels.npy ⭐️
             if not os.path.exists(label_path):
                 label_path = os.path.normpath(os.path.join(config.BASE_DIR, "..", "face_index.faiss.labels.npy"))

             if not os.path.exists(label_path):
                 logging.error(f"Faiss 레이블 파일 없음: {label_path}")
                 find_best_match_faiss.labels = np.array(["Error"])
             else:
                 find_best_match_faiss.labels = np.load(label_path)
                 logging.info(f"Faiss 레이블 로드 완료: {label_path}")

        labels = find_best_match_faiss.labels

        query_embedding = np.expand_dims(embedding.astype('float32'), axis=0)
        # L2 정규화 (ArcFace는 코사인 유사도 사용)
        faiss.normalize_L2(query_embedding)

        similarities, indices = faiss_index.search(query_embedding, 1)
        best_similarity = similarities[0][0]

        if best_similarity >= threshold:
            best_match_name = labels[indices[0][0]]
            return best_match_name, best_similarity
        return "Unknown", best_similarity
    except Exception as e:
        logging.error(f"Faiss 검색 중 오류 발생: {e}", exc_info=True)
        return "Unknown", 0.0


def log_violation(frame: np.ndarray, person_name: str, event_type: str, cam_id: int):
    try:
        now = datetime.datetime.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        safe_event_type = "".join(c for c in event_type if c.isalnum() or c in ('-'))

        # ⭐️ 로그 저장 경로를 config.py에서 가져오도록 수정 ⭐️
        log_folder = config.Paths.LOG_FOLDER
        image_filename = os.path.join(log_folder, f"{timestamp_str}_CAM{cam_id}_{person_name}_{safe_event_type}.jpg")

        # ⭐️ 이미지 저장 경로가 유효한지 확인 ⭐️
        if not os.path.exists(log_folder):
             os.makedirs(log_folder, exist_ok=True)

        cv2.imwrite(image_filename, frame)

        # --- [수정] config에서 CSV 파일 경로를 가져오도록 변경 ---
        log_filename = config.Paths.LOG_CSV
        log_entry = f"{now.strftime('%Y-%m-%d %H:%M:%S')},{person_name},{event_type},CAM-{cam_id},{image_filename}\n"

        # ⭐️ CSV 파일 헤더 쓰기 로직 개선 ⭐️
        file_exists = os.path.exists(log_filename)
        with open(log_filename, 'a', encoding='utf-8-sig', newline='') as f:
            if not file_exists:
                f.write("Timestamp,Person,Event,CameraID,EvidenceFile\n")
            f.write(log_entry)

        logging.info(f"[CAM-{cam_id}] 이벤트 기록 저장: {person_name} - {event_type}")
    except Exception as e:
        logging.error(f"로그 파일/이미지 저장 실패: {e}", exc_info=True)


def is_person_horizontal(keypoints: Keypoints, bbox_xyxy: Tuple) -> bool:
    """
    개선된 넘어짐 감지 로직
    """
    try:
        # Keypoints 객체가 비어있는지, data가 None인지 확인
        if keypoints is None or keypoints.data is None or len(keypoints.data) == 0:
            return False

        points = keypoints.xy[0].cpu().numpy()
        confidences = keypoints.conf[0].cpu().numpy()

        # 1. 기본 유효성 검사
        valid_points_mask = confidences > config.Thresholds.POSE_CONFIDENCE
        valid_points = points[valid_points_mask]
        if len(valid_points) < config.Thresholds.MIN_VISIBLE_KEYPOINTS:
            return False

        # 2. 키포인트 인덱스 정의 (COCO 포맷)
        upper_indices = [0, 5, 6, 7, 8]  # nose, left_shoulder, right_shoulder, left_elbow, right_elbow
        lower_indices = [11, 12, 13, 14, 15, 16]  # left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle
        center_indices = [5, 6, 11, 12]  # shoulders and hips

        # 3. 유효한 키포인트 추출 (마스크 사용)
        upper_mask = valid_points_mask[upper_indices]
        lower_mask = valid_points_mask[lower_indices]
        center_mask = valid_points_mask[center_indices]

        if not np.any(upper_mask) or not np.any(lower_mask) or not np.any(center_mask):
            return False

        valid_upper_points_data = points[upper_indices][upper_mask]
        valid_lower_points_data = points[lower_indices][lower_mask]
        valid_center_points_data = points[center_indices][center_mask]

        # 4. 다중 조건 검증
        fall_score = 0
        total_checks = 0

        # 조건 1: 상체-하체 위치 관계 (기존 로직 개선)
        avg_upper_y = np.mean(valid_upper_points_data[:, 1])
        avg_lower_y = np.mean(valid_lower_points_data[:, 1])

        if avg_upper_y >= avg_lower_y:
            fall_score += 1
        total_checks += 1

        # 조건 2: 키포인트 분포의 가로세로 비율
        kpt_min_x, kpt_min_y = np.min(valid_points, axis=0)
        kpt_max_x, kpt_max_y = np.max(valid_points, axis=0)
        kpt_width, kpt_height = kpt_max_x - kpt_min_x, kpt_max_y - kpt_min_y

        if kpt_height > 1e-5:
            kpt_aspect_ratio = kpt_width / kpt_height
            if kpt_aspect_ratio > config.Thresholds.FALL_ASPECT_RATIO:
                fall_score += 1
            total_checks += 1

        # 조건 3: 중심축의 기울기 분석
        if len(valid_center_points_data) >= 2:
            shoulder_points_data = points[[5, 6]][valid_points_mask[[5, 6]]]
            hip_points_data = points[[11, 12]][valid_points_mask[[11, 12]]]

            if len(shoulder_points_data) > 0 and len(hip_points_data) > 0:
                shoulder_center_y = np.mean(shoulder_points_data[:, 1])
                hip_center_y = np.mean(hip_points_data[:, 1])

                # 수직 중심축이 수평에 가까우면 넘어짐 가능성 높음
                vertical_ratio = abs(shoulder_center_y - hip_center_y) / (kpt_height + 1e-6) # 0으로 나누기 방지
                if vertical_ratio < config.Thresholds.FALL_VERTICAL_RATIO_THRESHOLD:
                    fall_score += 1
                total_checks += 1

        # 조건 4: 키포인트 밀도 분석
        if len(valid_points) >= 8:
            x_coords = valid_points[:, 0]
            y_coords = valid_points[:, 1]

            x_std = np.std(x_coords)
            y_std = np.std(y_coords)

            if y_std > 0 and x_std > y_std * config.Thresholds.FALL_HORIZONTAL_SPREAD_RATIO:
                fall_score += 1
            total_checks += 1

        # 5. 최종 판단: 다수 조건 만족 시 넘어짐으로 판단
        if total_checks == 0: return False # 유효한 검사가 없으면 넘어짐 아님
        return (fall_score / total_checks) >= config.Thresholds.FALL_SCORE_THRESHOLD

    except Exception as e:
        logging.warning(f"개선된 넘어짐 감지 함수 오류: {e}", exc_info=True)
        return False


def clip_bbox_xyxy(bbox_xyxy: Tuple, frame_w: int, frame_h: int) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_w, x2), min(frame_h, y2)
    if (x2 - x1) > 0 and (y2 - y1) > 0:
        return x1, y1, x2, y2
    return None
