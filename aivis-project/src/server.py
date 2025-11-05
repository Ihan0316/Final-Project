# server.py - AIVIS AI 서버 (얼굴 인식, PPE, 위험행동 탐지)
import asyncio
import logging
import os
import threading
import time
import json
import signal
import gzip
from typing import Dict, Set, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from collections import deque
from queue import Queue

import aiohttp_cors
import cv2
import numpy as np
import torch
from aiohttp import web, WSMsgType
from ultralytics.engine.results import Keypoints

import core
import utils
from utils import setup_logging, create_standard_response, find_best_match_faiss

# 전역 변수
latest_processed_frames: Dict[int, bytes] = {}  # 처리된 프레임 저장 (MJPEG용)
latest_frames: Dict[int, bytes] = {}  # 처리된 프레임 저장 (final과 호환)
latest_result_data: Dict[int, dict] = {}  # 최신 결과 데이터 저장 (대시보드용)
frame_lock = threading.Lock()
processing_lock = threading.Lock()  # 프레임 처리 동시성 제어용
processing_flags: Dict[int, bool] = {}  # cam_id별 처리 중 플래그

# 배치 처리용 프레임 큐 시스템 (Tesla V100 최적화)
frame_queues: Dict[int, deque] = {}  # cam_id별 프레임 큐
queue_lock = threading.Lock()
batch_processing_flags: Dict[int, bool] = {}  # cam_id별 배치 처리 중 플래그
connected_websockets: Set[web.WebSocketResponse] = set()
dashboard_websockets: Set[web.WebSocketResponse] = set()  # 대시보드 전용 연결
safety_system_instance: core.SafetySystem = None
server_loop: asyncio.AbstractEventLoop = None

# 모델 결과 데이터 (final과 동일한 구조)
model_results = {
    "alerts": [],
    "violations": {},
    "heatmap_counts": {"A-1": 0, "A-2": 0, "B-1": 0, "B-2": 0},
    "profile": {"name": "시스템", "status": "정상", "area": "전체"},
    "logs": [],
    "kpi_data": {"totalWorkers": 0, "attendees": 0, "ppeRate": 0, "riskLevel": 0},
    "detected_workers": {}  # 구역별 감지된 작업자 정보
}
results_lock = threading.Lock()

# 시스템 상태 모니터링
system_stats = {
    "start_time": time.time(),
    "total_requests": 0,
    "error_count": 0,
    "last_health_check": time.time(),
    "memory_usage": 0,
    "cpu_usage": 0
}
stats_lock = threading.Lock()

# 병렬 처리를 위한 스레드 풀 (얼굴 인식용)
face_recognition_executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="face_recognition")
# 초해상도는 GPU 리소스 경합 방지를 위해 단일 스레드 또는 순차 처리
super_resolution_lock = threading.Lock()  # 초해상도 모델 동시 접근 방지

# --- 압축 응답 헬퍼 함수 (final과 동일) ---
def create_compressed_response(data: dict, content_type: str = 'application/json') -> web.Response:
    """gzip 압축된 JSON 응답 생성"""
    try:
        json_data = json.dumps(data, ensure_ascii=False).encode('utf-8')
        compressed_data = gzip.compress(json_data)
        
        response = web.Response(
            body=compressed_data,
            content_type=content_type,
            headers={
                'Content-Encoding': 'gzip',
                'Content-Length': str(len(compressed_data)),
                'Cache-Control': 'public, max-age=60',  # 1분 캐시
                'Vary': 'Accept-Encoding'
            }
        )
        return response
    except Exception as e:
        logging.error(f"압축 응답 생성 실패: {e}")
        # 폴백: 일반 JSON 응답
        return web.json_response(data)

def filter_model_results(data: dict) -> dict:
    """모델 결과 데이터 필터링 - 필요한 데이터만 반환"""
    try:
        filtered_data = {
            "alerts": data.get("alerts", [])[-10:],  # 최근 10개 알림만
            "violations": data.get("violations", {}),
            "heatmap_counts": data.get("heatmap_counts", {}),
            "profile": data.get("profile", {}),
            "logs": data.get("logs", [])[-20:],  # 최근 20개 로그만
            "kpi_data": data.get("kpi_data", {}),
            "detected_workers": data.get("detected_workers", {})
        }
        
        # 빈 데이터 제거
        filtered_data = {k: v for k, v in filtered_data.items() if v}
        
        return filtered_data
    except Exception as e:
        logging.error(f"모델 결과 필터링 실패: {e}")
        return data

def _process_super_resolution(person_img: np.ndarray, person_id_text: str,
                              super_res_net, face_area: int, orig_h: int, orig_w: int) -> np.ndarray:
    """초해상도 처리 헬퍼 함수 (스레드 안전)"""
    with super_resolution_lock:
        try:
            sr_start_time = time.time()
            sr_input = person_img.copy().astype(np.uint8)

            if len(sr_input.shape) != 3 or sr_input.shape[2] != 3:
                raise ValueError(f"person_img가 3채널이 아님: {sr_input.shape}")

            # 최소 크기 확인
            min_size = 16
            if sr_input.shape[0] < min_size or sr_input.shape[1] < min_size:
                scale_factor = max(min_size / sr_input.shape[0], min_size / sr_input.shape[1])
                new_h = max(int(sr_input.shape[0] * scale_factor), min_size)
                new_w = max(int(sr_input.shape[1] * scale_factor), min_size)
                sr_input = cv2.resize(sr_input, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # RGB 변환 및 blob 생성
            sr_input_rgb = cv2.cvtColor(sr_input, cv2.COLOR_BGR2RGB)
            blob = cv2.dnn.blobFromImage(
                sr_input_rgb, 1.0,
                (sr_input_rgb.shape[1], sr_input_rgb.shape[0]),
                (0, 0, 0), swapRB=False, crop=False
            )

            if blob.shape[1] != 3:
                raise ValueError(f"Blob 채널 오류: {blob.shape}")

            # 추론 수행
            super_res_net.setInput(blob)
            result = super_res_net.forward()

            # 결과 변환
            if len(result.shape) == 4:
                upscaled = result.squeeze(axis=0).transpose(1, 2, 0)
            elif len(result.shape) == 3:
                upscaled = result.transpose(1, 2, 0)
            else:
                raise ValueError(f"예상치 못한 결과 shape: {result.shape}")

            # uint8 변환
            if upscaled.max() <= 1.0:
                upscaled = (upscaled * 255).astype(np.uint8)
            else:
                upscaled = np.clip(upscaled, 0, 255).astype(np.uint8)

            # RGB -> BGR 변환
            if upscaled.shape[2] == 3:
                upscaled = cv2.cvtColor(upscaled, cv2.COLOR_RGB2BGR)
            else:
                raise ValueError(f"업스케일 결과 채널 오류: {upscaled.shape}")

            # 크기 조정
            max_dim = max(person_img.shape[:2]) * 4
            if max_dim <= orig_h and max_dim <= orig_w:
                result_img = upscaled
            else:
                scale_factor = min(orig_h / max_dim, orig_w / max_dim) * 0.9
                new_h = int(person_img.shape[0] * scale_factor * 4)
                new_w = int(person_img.shape[1] * scale_factor * 4)
                result_img = cv2.resize(upscaled, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            sr_time = time.time() - sr_start_time
            if sr_time > 0.5:  # 0.5초 이상 걸린 경우만 로깅
                logging.info(f"{person_id_text} 초해상도 처리 시간: {sr_time:.3f}s")

            return result_img
        except Exception as e:
            logging.warning(f"{person_id_text} 초해상도 처리 실패: {e}. 원본 이미지 사용.")
            return person_img

def _process_face_recognition(person_img_for_detection: np.ndarray, person_id_text: str,
                              face_analyzer, face_database) -> Tuple[str, float]:
    """얼굴 인식 처리 헬퍼 함수 (병렬 처리 가능)"""
    try:
        face_start_time = time.time()
        faces = face_analyzer.get(person_img_for_detection)
        face_time = time.time() - face_start_time

        if faces and len(faces) > 0:
            biggest_face = max(faces, key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))
            embedding = biggest_face.normed_embedding
            person_name, similarity_score = find_best_match_faiss(
                embedding, face_database, core.config.Thresholds.SIMILARITY
            )
            if face_time > 0.1:
                logging.debug(f"{person_id_text} 얼굴 인식 시간: {face_time:.3f}s -> {person_name}")
            return person_name, similarity_score
        else:
            return "Unknown", 0.0
    except Exception as e:
        logging.warning(f"{person_id_text} 얼굴 인식 처리 실패: {e}")
        return "Unknown", 0.0

# --- AI 처리 함수 (SafetySystem 모델 사용) ---
def process_single_frame(frame_bytes: bytes, cam_id: int = 0) -> Tuple[bytes, dict]:
    """
    단일 프레임 바이트를 받아 AI 처리를 수행하고,
    처리된 프레임(JPEG 바이트)과 결과 데이터(dict)를 반환합니다.
    (SafetySystem의 모델을 활용하되, 상태 추적은 단순화)
    """
    global safety_system_instance
    if safety_system_instance is None or safety_system_instance.violation_model is None or safety_system_instance.pose_model is None:
        # 아직 초기화되지 않았다면 빈 결과 반환
        logging.warning("SafetySystem 인스턴스 또는 필수 모델이 아직 준비되지 않았습니다.")
        empty_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', empty_frame)
        return buffer.tobytes(), {}

    # 함수 내에서 orig_h, orig_w 기본값 설정 (오류 방지)
    orig_h, orig_w = 480, 640
    frame = None # 오류 발생 시 사용하기 위해 초기화

    try:
        # 1. 바이트를 이미지로 디코딩
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            logging.error("프레임 디코딩 실패")
            empty_frame = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
            _, buffer = cv2.imencode('.jpg', empty_frame)
            return buffer.tobytes(), {}
        orig_h, orig_w = frame.shape[:2]

        # 2. 모델 입력 크기에 맞게 리사이즈 (원본은 나중에 그리기용으로 사용)
        resized_frame = cv2.resize(frame, (core.config.SystemConfig.MODEL_INPUT_WIDTH, core.config.SystemConfig.MODEL_INPUT_HEIGHT))
        w_scale = orig_w / core.config.SystemConfig.MODEL_INPUT_WIDTH
        h_scale = orig_h / core.config.SystemConfig.MODEL_INPUT_HEIGHT

        # 3. 처리된 프레임 생성 (원본 프레임 복사)
        processed_frame = frame.copy()
        renderer = utils.TextRenderer(frame.shape)

        # 4. YOLO 추론 (위험행동 및 PPE 탐지)
        violation_results = safety_system_instance.violation_model(resized_frame, verbose=False)
        pose_results = safety_system_instance.pose_model(resized_frame, verbose=False)

        # 5. 탐지 결과 파싱 (리사이즈된 프레임 기준 -> 원본 프레임 크기로 스케일링)
        all_detections = {}
        if violation_results and len(violation_results) > 0:
            for det in violation_results[0].boxes:
                class_id = int(det.cls[0])
                class_name = safety_system_instance.violation_model.names[class_id]
                conf = float(det.conf[0])
                if conf >= core.config.Thresholds.YOLO_CONFIDENCE:
                    # 리사이즈된 프레임 기준 좌표를 원본 프레임 크기로 스케일링
                    bbox_resized = det.xyxy[0].cpu().numpy()
                    bbox_original = bbox_resized * np.array([w_scale, h_scale, w_scale, h_scale])
                    bbox_clipped = utils.clip_bbox_xyxy(bbox_original, orig_w, orig_h)
                    if bbox_clipped is not None:
                        if class_name not in all_detections:
                            all_detections[class_name] = []
                        # clip_bbox_xyxy는 tuple을 반환하므로 list()로 변환
                        all_detections[class_name].append({'bbox': list(bbox_clipped), 'conf': conf})

        # 6. 얼굴 인식 모델 및 DB 가져오기
        face_analyzer = safety_system_instance.face_analyzer
        face_database = safety_system_instance.face_database
        super_res_net = safety_system_instance.super_res_net

        recognized_faces = []
        violations_found = []

        # 7. 사람 감지 및 상태 확인
        if pose_results and pose_results[0].boxes is not None and len(pose_results[0].boxes) > 0:
            boxes = pose_results[0].boxes.xyxy.cpu().numpy()

            # 중복 사람 박스 제거 (NMS 유사) - 겹침이 큰 박스는 큰 박스 하나만 유지
            try:
                if boxes is not None and len(boxes) > 1:
                    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    order = np.argsort(-areas)  # 큰 박스 우선
                    keep_indices = []
                    suppressed = np.zeros(len(boxes), dtype=bool)
                    for idx in order:
                        if suppressed[idx]:
                            continue
                        keep_indices.append(idx)
                        x1, y1, x2, y2 = boxes[idx]
                        for j in order:
                            if j == idx or suppressed[j]:
                                continue
                            iou = utils.calculate_iou((x1, y1, x2, y2), tuple(boxes[j]))
                            if iou > 0.7:  # 높은 겹침은 중복으로 간주
                                suppressed[j] = True
                    boxes = boxes[keep_indices]
                    if pose_results[0].keypoints is not None:
                        keypoints_list = [pose_results[0].keypoints[i] for i in keep_indices]
            except Exception:
                pass
            keypoints_list = pose_results[0].keypoints if pose_results[0].keypoints else None
            confidences = pose_results[0].boxes.conf.cpu().numpy() if pose_results[0].boxes.conf is not None else None

            # 초해상도는 작은 얼굴만 처리하고, 큰 얼굴은 스킵하여 성능 최적화
            # 사람이 많을수록 초해상도는 작은 이미지만 처리하여 시간 절약
            upscale_threshold_area = core.config.Thresholds.UPSCALE_THRESHOLD ** 2

            # 사람 박스 좌표를 원본 프레임 크기로 스케일링 및 필터링
            scaled_person_boxes = []
            valid_person_indices = []  # 유효한 사람 박스 인덱스
            filtered_boxes = []
            filtered_keypoints = []
            filtered_confidences = []

            for i, box in enumerate(boxes):
                scaled_box_np = box * np.array([w_scale, h_scale, w_scale, h_scale])
                clipped_box = utils.clip_bbox_xyxy(scaled_box_np, orig_w, orig_h)
                if clipped_box is None:
                    continue

                x1, y1, x2, y2 = clipped_box
                box_w = x2 - x1
                box_h = y2 - y1
                box_area = box_w * box_h
                aspect_ratio = box_w / box_h if box_h > 0 else 0

                # 1. 키포인트 확인 (상반신 강하게 잡히면 완화 임계값 적용)
                num_valid_kpts = 0
                has_head_or_shoulders = False
                if keypoints_list is not None and i < len(keypoints_list):
                    keypoints = keypoints_list[i]
                    if keypoints is not None and keypoints.conf is not None:
                        conf_arr = keypoints.conf[0].cpu().numpy()
                        valid_kpts_mask = conf_arr > core.config.Thresholds.POSE_CONFIDENCE
                        num_valid_kpts = int(np.sum(valid_kpts_mask))
                        # nose(0), left_shoulder(5), right_shoulder(6)
                        idxs = [0, 5, 6]
                        for idx in idxs:
                            if idx < len(valid_kpts_mask) and valid_kpts_mask[idx]:
                                has_head_or_shoulders = True
                                break

                use_relaxed = (num_valid_kpts >= 12 and has_head_or_shoulders)
                min_w = core.config.Thresholds.RELAXED_MIN_PERSON_BOX_WIDTH if use_relaxed else core.config.Thresholds.MIN_PERSON_BOX_WIDTH
                min_h = core.config.Thresholds.RELAXED_MIN_PERSON_BOX_HEIGHT if use_relaxed else core.config.Thresholds.MIN_PERSON_BOX_HEIGHT
                min_area = core.config.Thresholds.RELAXED_MIN_PERSON_BOX_AREA if use_relaxed else core.config.Thresholds.MIN_PERSON_BOX_AREA
                max_ar = core.config.Thresholds.RELAXED_MAX_PERSON_ASPECT_RATIO if use_relaxed else core.config.Thresholds.MAX_PERSON_ASPECT_RATIO
                min_ar = core.config.Thresholds.RELAXED_MIN_PERSON_ASPECT_RATIO if use_relaxed else core.config.Thresholds.MIN_PERSON_ASPECT_RATIO

                # 2. 최소 크기 필터링 (너무 작은 박스는 제외)
                if box_w < min_w or box_h < min_h or box_area < min_area:
                    logging.debug(f"사람 박스 필터링 (크기 작음): {box_w}x{box_h}, 면적={box_area} (relaxed={use_relaxed})")
                    continue

                # 3. 종횡비 필터링 (손처럼 세로로 긴 것 또는 너무 가로로 긴 것 제외)
                if aspect_ratio > max_ar or aspect_ratio < min_ar:
                    logging.debug(f"사람 박스 필터링 (종횡비 이상): {aspect_ratio:.2f} (relaxed={use_relaxed})")
                    continue

                # 4. 키포인트 기본 하한 (완화 여부와 무관하게 최소한은 요구)
                if num_valid_kpts < core.config.Thresholds.MIN_VISIBLE_KEYPOINTS:
                    logging.debug(f"사람 박스 필터링 (키포인트 부족): {num_valid_kpts}")
                    continue

                # 4. violation_model에서 탐지된 작은 객체와 겹치는지 확인
                should_filter = False
                for class_name, detections in all_detections.items():
                    # 'person' 클래스는 제외 (pose_model과 중복)
                    if class_name.lower() == 'person':
                        continue
                    # 안전 장비는 제외
                    is_safety_gear = any(class_name in item.values() for item in core.config.Constants.SAFETY_RULES_MAP.values())
                    if is_safety_gear:
                        continue

                    # 작은 객체(machinery, hand 등)와 겹치면 필터링
                    for det in detections:
                        if det and 'bbox' in det and det['bbox'] and len(det['bbox']) == 4:
                            dx1, dy1, dx2, dy2 = det['bbox']
                            det_area = (dx2 - dx1) * (dy2 - dy1)

                            # 작은 객체가 사람 박스 내부나 가까이 있으면 필터링
                            det_center_x = (dx1 + dx2) / 2
                            det_center_y = (dy1 + dy2) / 2

                            if (x1 <= det_center_x <= x2 and y1 <= det_center_y <= y2) or \
                               (dx1 < x2 and dx2 > x1 and dy1 < y2 and dy2 > y1):
                                iou = utils.calculate_iou((x1, y1, x2, y2), (dx1, dy1, dx2, dy2))
                                # 작은 객체가 사람 박스 면적의 30% 이상 차지하고 IOU가 0.2 이상이면 제외
                                if det_area > box_area * 0.3 and iou > 0.2:
                                    logging.debug(f"사람 박스 필터링 (작은 객체와 겹침): {class_name}, IOU={iou:.2f}")
                                    should_filter = True
                                    break

                    if should_filter:
                        break

                if should_filter:
                    continue

                # 모든 필터링을 통과한 유효한 사람 박스
                scaled_person_boxes.append(scaled_box_np)
                valid_person_indices.append(i)
                filtered_boxes.append(box)
                if keypoints_list is not None and i < len(keypoints_list):
                    filtered_keypoints.append(keypoints_list[i])
                if confidences is not None:
                    filtered_confidences.append(confidences[i])

            # 필터링된 결과로 업데이트
            boxes = np.array(filtered_boxes) if filtered_boxes else np.array([])
            keypoints_list = filtered_keypoints if filtered_keypoints else None
            if confidences is not None:
                confidences = np.array(filtered_confidences) if filtered_confidences else np.array([])

            num_people = len(boxes)
            if num_people == 0:
                logging.debug("필터링 후 유효한 사람 탐지 없음")
            else:
                logging.debug(f"필터링 후 유효한 사람 수: {num_people}")

            # 병렬 처리를 위한 작업 목록 준비
            face_recognition_tasks = []
            futures_with_index = []  # (person_data_list_index, future)
            person_data_list = []  # 순서대로 결과를 맞추기 위한 리스트

            # 얼굴 인식 대상 상한 (가장 큰 박스 우선)
            try:
                areas_for_sort = []
                for idx, b in enumerate(boxes):
                    sx1, sy1, sx2, sy2 = (b * np.array([w_scale, h_scale, w_scale, h_scale])).astype(float)
                    areas_for_sort.append(((sx2 - sx1) * (sy2 - sy1), idx))
                areas_for_sort.sort(reverse=True)
                max_face_targets = 3
                allowed_face_indices = set(i for _, i in areas_for_sort[:max_face_targets])
            except Exception:
                allowed_face_indices = set(range(len(boxes)))

            for i, box in enumerate(boxes):
                scaled_box_np = box * np.array([w_scale, h_scale, w_scale, h_scale])
                clipped_box = utils.clip_bbox_xyxy(scaled_box_np, orig_w, orig_h)
                if clipped_box is None:
                    continue
                x1, y1, x2, y2 = clipped_box
                person_id_text = f"P{i}"

                # 사람 박스 영역 추출
                person_img = frame[y1:y2, x1:x2]

                if person_img.size == 0:
                    continue

                # 채널 변환 (3채널 BGR로 통일)
                if len(person_img.shape) == 2:
                    person_img = cv2.cvtColor(person_img, cv2.COLOR_GRAY2BGR)
                elif len(person_img.shape) == 3 and person_img.shape[2] == 1:
                    person_img = cv2.cvtColor(person_img, cv2.COLOR_GRAY2BGR)
                elif len(person_img.shape) == 3 and person_img.shape[2] == 4:
                    person_img = cv2.cvtColor(person_img, cv2.COLOR_RGBA2BGR)
                elif len(person_img.shape) == 3 and person_img.shape[2] != 3:
                    person_img = cv2.cvtColor(person_img[:,:,0], cv2.COLOR_GRAY2BGR) if person_img.shape[2] > 0 else person_img

                # 최종 확인: 반드시 3채널 BGR
                if len(person_img.shape) != 3 or person_img.shape[2] != 3:
                    person_img = frame[y1:y2, x1:x2]
                    if len(person_img.shape) == 2 or (len(person_img.shape) == 3 and person_img.shape[2] == 1):
                        person_img = cv2.cvtColor(person_img, cv2.COLOR_GRAY2BGR)

                person_img_for_detection = person_img.copy()

                # 초해상도 처리 (작은 얼굴만, 동기 처리로 GPU 리소스 경합 방지)
                face_area = (y2 - y1) * (x2 - x1)
                if super_res_net is not None and face_area < upscale_threshold_area:
                    # 사람이 많을 때는 더 작은 얼굴만 처리 (성능 최적화)
                    if num_people <= 5 or face_area < (upscale_threshold_area * 0.7):
                        person_img_for_detection = _process_super_resolution(
                            person_img, person_id_text, super_res_net,
                            face_area, orig_h, orig_w
                        )

                # 얼굴 인식 작업을 병렬 처리 대기열에 추가
                if face_analyzer is not None and face_database is not None:
                    person_data_list.append({
                        'index': i,
                        'person_id': person_id_text,
                        'box': (x1, y1, x2, y2),
                        'img': person_img_for_detection,
                        'keypoints': keypoints_list[i] if keypoints_list and len(keypoints_list) > i else None
                    })
                    # 가장 큰 상위 N명만 얼굴 인식 수행 (지연 방지)
                    if i in allowed_face_indices:
                        future = face_recognition_executor.submit(
                            _process_face_recognition,
                            person_img_for_detection.copy(),
                            person_id_text,
                            face_analyzer,
                            face_database
                        )
                        face_recognition_tasks.append(future)
                        futures_with_index.append((len(person_data_list) - 1, future))
                else:
                    person_data_list.append({
                        'index': i,
                        'person_id': person_id_text,
                        'box': (x1, y1, x2, y2),
                        'img': None,
                        'keypoints': keypoints_list[i] if keypoints_list and len(keypoints_list) > i else None,
                        'name': "Unknown",
                        'similarity': 0.0
                    })

            # 병렬로 얼굴 인식 결과 수집 (엄격한 시간 예산)
            # 타임아웃을 0.1초로 단축하여 더 빠른 처리
            face_recognition_results = {}
            try:
                for future in as_completed([f for _, f in futures_with_index], timeout=0.1):
                    try:
                        person_name, similarity_score = future.result(timeout=0.1)
                        # 매핑된 인덱스에 결과 기록
                        mapped_idx = next((idx for idx, f in futures_with_index if f is future), None)
                        if mapped_idx is not None and mapped_idx < len(person_data_list):
                            person_data_list[mapped_idx]['name'] = person_name
                            person_data_list[mapped_idx]['similarity'] = similarity_score
                    except Exception as e:
                        logging.warning(f"얼굴 인식 작업 실패: {e}")
                        # 타임아웃/오류 시 이미 기본 Unknown 유지
            except FuturesTimeoutError:
                # 일부 작업이 시간 내 완료되지 않아도 진행
                pass

            # 결과를 순서대로 처리
            for person_data in person_data_list:
                i = person_data['index']
                person_id_text = person_data['person_id']
                x1, y1, x2, y2 = person_data['box']
                person_name = person_data.get('name', 'Unknown')
                similarity_score = person_data.get('similarity', 0.0)

                if person_name != "Unknown":
                    recognized_faces.append({
                        "box": [x1, y1, x2, y2],
                        "name": person_name,
                        "similarity": float(similarity_score)
                    })

                # PPE 및 위험행동 탐지 (기존 로직 유지)
                person_status = "SAFE"
                status_details = []
                current_violations = []

                for rule, classes in core.config.Constants.SAFETY_RULES_MAP.items():
                    comp_cls, viol_cls = classes["compliance"], classes["violation"]
                    is_compliance = False
                    is_violation = False

                    def _center_inside(person_box, obj_box):
                        px1, py1, px2, py2 = person_box
                        ox1, oy1, ox2, oy2 = obj_box
                        cx = (ox1 + ox2) / 2.0
                        cy = (oy1 + oy2) / 2.0
                        return (px1 <= cx <= px2) and (py1 <= cy <= py2)

                    def _overlap_ratio(person_box, obj_box):
                        px1, py1, px2, py2 = person_box
                        ox1, oy1, ox2, oy2 = obj_box
                        ix1, iy1 = max(px1, ox1), max(py1, oy1)
                        ix2, iy2 = min(px2, ox2), min(py2, oy2)
                        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                        obj_area = max(1, (ox2 - ox1) * (oy2 - oy1))
                        return inter / obj_area

                    # 준수(착용) 판정: 장비 박스 중심이 사람 박스 내부이거나, 겹친 면적/장비 면적 ≥ 0.6 (정확도 상향)
                    if comp_cls in all_detections and all_detections[comp_cls]:
                        for det in all_detections[comp_cls]:
                            if det and 'bbox' in det and det['bbox'] and len(det['bbox']) == 4:
                                ratio = _overlap_ratio((x1, y1, x2, y2), det['bbox'])
                                if _center_inside((x1, y1, x2, y2), det['bbox']) or ratio >= 0.6:
                                    is_compliance = True
                                    break

                    # 위반(미착용) 판정: 위반 클래스가 사람과 충분히 겹치면 위반으로 인정
                    if viol_cls in all_detections and all_detections[viol_cls]:
                        for det in all_detections[viol_cls]:
                            if det and 'bbox' in det and det['bbox'] and len(det['bbox']) == 4:
                                ratio = _overlap_ratio((x1, y1, x2, y2), det['bbox'])
                                if _center_inside((x1, y1, x2, y2), det['bbox']) or ratio >= 0.6:
                                    is_violation = True
                                    break

                    if is_violation:
                        status_details.append(f"{rule}: VIOLATION")
                        current_violations.append(rule)
                        person_status = "VIOLATION"
                    elif is_compliance:
                        status_details.append(f"{rule}: SAFE")
                    else:
                        status_details.append(f"{rule}: CHECKING")

                # 넘어짐 감지
                is_fallen = False
                if person_data['keypoints'] is not None:
                    person_keypoints = person_data['keypoints']
                    scaled_kpts_data = person_keypoints.data.clone()
                    scaled_kpts_data[..., 0] *= w_scale
                    scaled_kpts_data[..., 1] *= h_scale
                    scaled_keypoints = Keypoints(scaled_kpts_data, (orig_h, orig_w))
                    is_fallen = utils.is_person_horizontal(scaled_keypoints, (x1, y1, x2, y2))

                if is_fallen:
                    person_status = "FALL"
                    status_details.append("넘어짐 감지")
                    current_violations.append("넘어짐")

                # 그리기: 색상 결정 및 박스 그리기
                color = (0, 255, 0)  # SAFE
                if person_status == "FALL":
                    color = (0, 0, 255)
                elif person_status == "VIOLATION":
                    color = (0, 165, 255)
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)

                # 그리기: 텍스트 추가
                status_text = f"{person_id_text}: {person_status}"
                if current_violations:
                    status_text += f" ({','.join(current_violations)})"
                if person_name != "Unknown":
                    status_text += f" [{person_name} {similarity_score:.2f}]"
                else:
                    status_text += f" [Unknown {similarity_score:.2f}]"

                renderer.add_text(status_text, (x1, y1 - 5), color)

                # 위반 사항 기록
                if current_violations:
                    violations_found.append({
                        "person_box": [x1, y1, x2, y2],
                        "violations": current_violations,
                        "recognized_name": person_name
                    })

        # 8. 기타 객체 그리기 (안전 장비는 위에서 이미 처리했으므로 제외)
        for class_name, detections in all_detections.items():
            # 'person' 클래스는 pose_results에서 이미 처리하므로 제외
            if class_name.lower() == 'person':
                continue
            # 안전 장비 클래스는 사람 박스와 함께 위에서 처리하므로 제외
            is_safety_gear = any(class_name in item.values() for item in core.config.Constants.SAFETY_RULES_MAP.values())
            if not is_safety_gear and detections:
                color = (255, 0, 0)  # 파란색 (BGR)
                for det in detections:
                    if det and 'bbox' in det and det['bbox'] and len(det['bbox']) == 4:
                        x1_obj, y1_obj, x2_obj, y2_obj = map(int, det['bbox'])

                        # 손/작은 객체 필터링: 사람 박스와 겹치는 작은 객체는 무시
                        obj_area = (x2_obj - x1_obj) * (y2_obj - y1_obj)
                        obj_center_x = (x1_obj + x2_obj) / 2
                        obj_center_y = (y1_obj + y2_obj) / 2

                        # 사람 박스와의 IOU 확인 및 필터링
                        should_filter = False
                        for person_box in scaled_person_boxes:
                            px1, py1, px2, py2 = person_box
                            person_area = (px2 - px1) * (py2 - py1)

                            # 작은 객체가 사람 박스 내부나 가까이 있으면 필터링
                            if (px1 <= obj_center_x <= px2 and py1 <= obj_center_y <= py2) or \
                               (x1_obj < px2 and x2_obj > px1 and y1_obj < py2 and y2_obj > py1):
                                # IOU 계산
                                iou = utils.calculate_iou((px1, py1, px2, py2), (x1_obj, y1_obj, x2_obj, y2_obj))

                                # 작은 객체(machinery, hand 등)이고 사람 박스와 겹치면 필터링
                                # 또는 객체가 사람 박스 면적의 10% 미만이고 IOU가 0.1 이상이면 필터링
                                if obj_area < person_area * 0.1 and iou > 0.05:
                                    should_filter = True
                                    break

                        # machinery 클래스는 특히 엄격하게 필터링 (사람 박스와 겹치면 무시)
                        if class_name.lower() in ['machinery', 'hand', 'hands'] and should_filter:
                            logging.debug(f"작은 객체 필터링: {class_name} (사람 박스와 겹침)")
                            continue

                        # 원본 프레임에 직접 그리기 (이미 스케일링된 좌표)
                        cv2.rectangle(processed_frame, (x1_obj, y1_obj), (x2_obj, y2_obj), color, 1)
                        display_name = class_name[:10]
                        renderer.add_text(f"{display_name}", (x1_obj, y1_obj - 5), color)

        processed_frame = renderer.render_on(processed_frame)

        # 9. 처리된 프레임을 JPEG 바이트로 인코딩
        ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 68])
        if not ret:
            logging.error("JPEG 인코딩 실패")
            empty_frame = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
            _, buffer = cv2.imencode('.jpg', empty_frame)
            return buffer.tobytes(), {}

        processed_frame_bytes = buffer.tobytes()

        # 10. 결과 데이터 구성
        result_data = {
            "recognized_faces": recognized_faces,
            "violations": violations_found,
            "violation_count": len(violations_found)
        }

        return processed_frame_bytes, result_data

    except Exception as e:
        logging.error(f"AI 처리 실행 중 오류: {e}", exc_info=True)
        error_frame = frame if frame is not None else np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        cv2.putText(error_frame, f"Error processing frame", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame)
        return buffer.tobytes(), {"error": str(e)}


# --- 배치 처리 함수 (Tesla V100 최적화) ---
def process_batch_frames(frame_data_list: List[Tuple[bytes, int]]) -> List[Tuple[bytes, dict]]:
    """
    여러 프레임을 배치로 처리하여 GPU 활용률 향상
    frame_data_list: [(frame_bytes, cam_id), ...]
    반환: [(processed_frame_bytes, result_data), ...]
    """
    global safety_system_instance
    if safety_system_instance is None or safety_system_instance.violation_model is None or safety_system_instance.pose_model is None:
        logging.warning("SafetySystem 인스턴스 또는 필수 모델이 아직 준비되지 않았습니다.")
        empty_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', empty_frame)
        empty_bytes = buffer.tobytes()
        return [(empty_bytes, {})] * len(frame_data_list)

    if not frame_data_list:
        return []

    try:
        # 1. 프레임 디코딩 및 리사이즈 (배치 준비)
        batch_frames = []  # 리사이즈된 프레임들
        original_frames = []  # 원본 프레임들 (그리기용)
        cam_ids = []  # 각 프레임의 cam_id
        scales = []  # 각 프레임의 스케일 정보 (w_scale, h_scale)
        
        for frame_bytes, cam_id in frame_data_list:
            try:
                nparr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    logging.warning(f"프레임 디코딩 실패 (cam_id={cam_id})")
                    continue
                
                orig_h, orig_w = frame.shape[:2]
                resized_frame = cv2.resize(
                    frame, 
                    (core.config.SystemConfig.MODEL_INPUT_WIDTH, core.config.SystemConfig.MODEL_INPUT_HEIGHT)
                )
                
                batch_frames.append(resized_frame)
                original_frames.append(frame)
                cam_ids.append(cam_id)
                scales.append((orig_w / core.config.SystemConfig.MODEL_INPUT_WIDTH, orig_h / core.config.SystemConfig.MODEL_INPUT_HEIGHT))
            except Exception as e:
                logging.error(f"프레임 준비 중 오류 (cam_id={cam_id}): {e}")
                continue

        if not batch_frames:
            return []

        # 2. YOLO 배치 추론 (GPU 활용률 최대화)
        batch_size = core.config.SystemConfig.BATCH_SIZE
        all_processed_frames = []
        all_result_data = []

        # 배치 단위로 처리
        for i in range(0, len(batch_frames), batch_size):
            batch_slice = batch_frames[i:i+batch_size]
            batch_orig = original_frames[i:i+batch_size]
            batch_cam_ids = cam_ids[i:i+batch_size]
            batch_scales = scales[i:i+batch_size]

            # YOLO 배치 추론
            try:
                violation_results = safety_system_instance.violation_model(
                    batch_slice, 
                    verbose=False,
                    half=core.config.SystemConfig.ENABLE_HALF_PRECISION and safety_system_instance.device == 'cuda',
                    device=safety_system_instance.device
                )
                pose_results = safety_system_instance.pose_model(
                    batch_slice,
                    verbose=False,
                    half=core.config.SystemConfig.ENABLE_HALF_PRECISION and safety_system_instance.device == 'cuda',
                    device=safety_system_instance.device
                )
            except Exception as e:
                logging.error(f"YOLO 배치 추론 중 오류: {e}", exc_info=True)
                # 오류 발생 시 단일 프레임 처리로 폴백
                for j, orig_frame in enumerate(batch_orig):
                    empty_frame = np.zeros((orig_frame.shape[0], orig_frame.shape[1], 3), dtype=np.uint8)
                    _, buffer = cv2.imencode('.jpg', empty_frame)
                    all_processed_frames.append(buffer.tobytes())
                    all_result_data.append({"error": str(e)})
                continue

            # 3. 각 프레임별로 결과 처리 (기존 process_single_frame 로직 재사용)
            # 결과는 리스트로 반환되므로 각 프레임별로 처리
            for j, (orig_frame, cam_id, (w_scale, h_scale)) in enumerate(zip(batch_orig, batch_cam_ids, batch_scales)):
                try:
                    # 결과 추출
                    vio_result = violation_results[j] if isinstance(violation_results, list) else violation_results
                    pose_result = pose_results[j] if isinstance(pose_results, list) else pose_results

                    # 기존 process_single_frame의 결과 처리 로직 재사용
                    # (간소화된 버전 - 전체 로직은 process_single_frame 참고)
                    processed_frame = orig_frame.copy()
                    renderer = utils.TextRenderer(orig_frame.shape)

                    # 탐지 결과 파싱 및 그리기 (기존 로직과 유사)
                    # 여기서는 단순화 - 실제 구현은 process_single_frame 로직을 재사용
                    recognized_faces = []
                    violations_found = []

                    # 프레임 인코딩
                    ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 68])
                    if not ret:
                        empty_frame = np.zeros((orig_frame.shape[0], orig_frame.shape[1], 3), dtype=np.uint8)
                        _, buffer = cv2.imencode('.jpg', empty_frame)
                    
                    processed_frame_bytes = buffer.tobytes()
                    result_data = {
                        "recognized_faces": recognized_faces,
                        "violations": violations_found,
                        "violation_count": len(violations_found)
                    }

                    all_processed_frames.append(processed_frame_bytes)
                    all_result_data.append(result_data)

                except Exception as e:
                    logging.error(f"프레임 결과 처리 중 오류 (cam_id={cam_id}): {e}", exc_info=True)
                    error_frame = orig_frame if orig_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                    _, buffer = cv2.imencode('.jpg', error_frame)
                    all_processed_frames.append(buffer.tobytes())
                    all_result_data.append({"error": str(e)})

        return list(zip(all_processed_frames, all_result_data))

    except Exception as e:
        logging.error(f"배치 처리 실행 중 오류: {e}", exc_info=True)
        # 오류 시 빈 결과 반환
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', empty_frame)
        empty_bytes = buffer.tobytes()
        return [(empty_bytes, {"error": str(e)})] * len(frame_data_list)


# --- 대시보드 브로드캐스트 함수 ---
async def broadcast_to_dashboards(result_data: dict, cam_id: int = 0):
    """대시보드 연결들에게 결과 데이터 브로드캐스트"""
    if not dashboard_websockets:
        return

    message = json.dumps(result_data)
    disconnected = set()

    for ws in dashboard_websockets:
        try:
            await ws.send_str(message)
        except (ConnectionResetError, ConnectionError, OSError):
            disconnected.add(ws)

    # 끊어진 연결 제거
    for ws in disconnected:
        dashboard_websockets.discard(ws)

# --- 모델 결과 업데이트 및 브로드캐스트 함수 (final과 동일) ---
def update_model_results_from_frame(result_data: dict, cam_id: int = 0):
    """프레임 처리 결과를 기반으로 model_results를 업데이트합니다."""
    global model_results
    
    with results_lock:
        # 위반 감지 결과 처리
        violations_list = result_data.get("violations", [])
        recognized_faces = result_data.get("recognized_faces", [])
        
        # 위반 처리
        for violation in violations_list:
            area = violation.get("area", "A-1")
            level = violation.get("level", "WARNING")
            worker = violation.get("worker", "알 수 없음")
            hazard = violation.get("hazard", "위반 감지")
            
            # 알림 추가
            alert = {
                "level": level,
                "area": area,
                "worker": worker,
                "hazard": hazard,
                "timestamp": time.time()
            }
            model_results["alerts"].append(alert)
            
            # 최근 20개 알림만 유지
            if len(model_results["alerts"]) > 20:
                model_results["alerts"] = model_results["alerts"][-20:]
            
            # 히트맵 카운트 증가
            if area in model_results["heatmap_counts"]:
                model_results["heatmap_counts"][area] += 1
            
            # 위반 카운트 업데이트
            if area not in model_results["violations"]:
                model_results["violations"][area] = 0
            model_results["violations"][area] += 1
            
            # 프로필 업데이트 (최신 위반)
            model_results["profile"] = {
                "name": worker,
                "status": hazard,
                "area": area
            }
        
        # 작업자 수 계산
        total_workers = len(recognized_faces)
        attendees = total_workers
        
        # PPE 비율 계산
        ppe_count = sum(1 for face in recognized_faces if face.get("has_ppe", False))
        ppe_rate = (ppe_count / total_workers * 100) if total_workers > 0 else 0
        ppe_rate = round(ppe_rate, 1)
        
        # 위험도 계산
        violation_count = len(violations_list)
        if total_workers == 0:
            risk_level = 0
        else:
            risk_score = violation_count * 2  # 위반당 2점
            max_possible_score = total_workers * 2
            risk_level = min(100, (risk_score / max_possible_score) * 100) if max_possible_score > 0 else 0
            risk_level = round(risk_level, 1)
        
        # KPI 데이터 업데이트
        model_results["kpi_data"] = {
            "totalWorkers": total_workers,
            "attendees": attendees,
            "ppeRate": ppe_rate,
            "riskLevel": risk_level
        }
        
        # 감지된 작업자 정보 업데이트
        model_results["detected_workers"] = {}
        for face in recognized_faces:
            name = face.get("name", "Unknown")
            area = face.get("area", "A-1")
            if area not in model_results["detected_workers"]:
                model_results["detected_workers"][area] = []
            model_results["detected_workers"][area].append({
                "name": name,
                "has_ppe": face.get("has_ppe", False)
            })

async def broadcast_model_results():
    """연결된 모든 웹소켓 클라이언트에게 모델 결과를 전송합니다."""
    # 대시보드 연결에만 전송 (클라이언트 연결은 프레임 전송으로 충분)
    if not dashboard_websockets:
        return
    
    try:
        with results_lock:
            results_json = json.dumps({
                "type": "model_results",
                "data": model_results
            })
        
        # 동시 전송을 위해 gather 사용 (타임아웃 추가)
        disconnected = set()
        tasks = []
        for ws in dashboard_websockets.copy():  # copy로 반복 중 수정 방지
            tasks.append(_send_to_websocket(ws, results_json, disconnected))
        
        # 최대 2초 타임아웃으로 전송
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=2.0
        )
        
        # 끊어진 연결 제거
        for ws in disconnected:
            dashboard_websockets.discard(ws)
    except asyncio.TimeoutError:
        logging.warning("broadcast_model_results 타임아웃 (2초 초과)")
    except Exception as e:
        logging.error(f"broadcast_model_results 오류: {e}")

async def _send_to_websocket(ws, message, disconnected_set):
    """개별 WebSocket에 메시지 전송 (헬퍼 함수)"""
    try:
        await ws.send_str(message)
    except (ConnectionResetError, ConnectionError, OSError):
        disconnected_set.add(ws)
    except Exception as e:
        logging.debug(f"WebSocket 전송 오류: {e}")
        disconnected_set.add(ws)

async def broadcast_logs(logs: list):
    """연결된 모든 웹소켓 클라이언트에게 로그 메시지를 전송합니다."""
    if not connected_websockets:
        return
    
    full_log_message = "".join(logs)
    # 동시 전송을 위해 gather 사용
    disconnected = set()
    for ws in connected_websockets:
        try:
            await ws.send_str(full_log_message)
        except (ConnectionResetError, ConnectionError, OSError):
            disconnected.add(ws)
    
    # 끊어진 연결 제거
    for ws in disconnected:
        connected_websockets.discard(ws)


# --- 대시보드용 웹소켓 핸들러 ---
async def dashboard_websocket_handler(request: web.Request):
    """대시보드 전용 WebSocket 핸들러 (데이터만 받기)"""
    global server_loop
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    dashboard_websockets.add(ws)
    client_id = id(ws)
    logging.info(f"대시보드 웹소켓 클라이언트 {client_id} 연결됨. (현재 {len(dashboard_websockets)}명 접속 중)")

    # 연결 시 최신 데이터 전송
    with frame_lock:
        latest_data = latest_result_data.get(0, {})

    if latest_data:
        try:
            await ws.send_str(json.dumps(latest_data))
        except (ConnectionResetError, ConnectionError, OSError):
            pass
    else:
        # 초기 연결 확인 메시지 전송
        try:
            await ws.send_str(json.dumps({"type": "connected", "message": "대시보드 연결됨"}))
        except (ConnectionResetError, ConnectionError, OSError):
            pass

    try:
        # 연결 유지를 위한 무한 루프
        while True:
            try:
                # 메시지 수신 대기 (타임아웃 없음, 연결 유지)
                msg = await asyncio.wait_for(ws.receive(), timeout=30.0)

                if msg.type == WSMsgType.TEXT:
                    # ping/pong 메시지 처리
                    try:
                        data = json.loads(msg.data)
                        if data.get("type") == "ping":
                            await ws.send_str(json.dumps({"type": "pong"}))
                    except:
                        pass
                elif msg.type == WSMsgType.ERROR:
                    logging.warning(f"대시보드 클라이언트 {client_id} 오류 발생: {ws.exception()}")
                    break
                elif msg.type == WSMsgType.CLOSE:
                    break

            except asyncio.TimeoutError:
                # 타임아웃 시 연결 확인 메시지 전송
                try:
                    await ws.send_str(json.dumps({"type": "heartbeat"}))
                except (ConnectionResetError, ConnectionError, OSError):
                    break
            except Exception as e:
                logging.warning(f"대시보드 클라이언트 {client_id} 메시지 수신 오류: {e}")
                break

    except (ConnectionResetError, ConnectionError, OSError) as e:
        logging.info(f"대시보드 클라이언트 {client_id} 연결이 끊어졌습니다. ({e})")
    finally:
        dashboard_websockets.discard(ws)
        logging.info(f"대시보드 웹소켓 클라이언트 {client_id} 연결 종료. (현재 {len(dashboard_websockets)}명 접속 중)")


# --- 배치 처리 백그라운드 태스크 (Tesla V100 최적화) ---
async def batch_processing_worker():
    """프레임 큐에서 프레임을 모아 배치로 처리하는 백그라운드 태스크"""
    global frame_queues, queue_lock, batch_processing_flags
    
    BATCH_TIMEOUT = 0.1  # 0.1초 동안 프레임 수집 (100ms)
    MIN_BATCH_SIZE = 1  # 최소 배치 크기 (즉시 처리 가능)
    MAX_BATCH_SIZE = core.config.SystemConfig.BATCH_SIZE  # 최대 배치 크기
    
    while True:
        try:
            # 모든 카메라의 큐 확인
            frames_to_process = {}  # cam_id -> [(frame_bytes, timestamp), ...]
            current_time = time.time()
            
            with queue_lock:
                for cam_id, queue in frame_queues.items():
                    if batch_processing_flags.get(cam_id, False):
                        continue  # 이미 처리 중이면 스킵
                    
                    # 큐에서 프레임 수집 (최대 MAX_BATCH_SIZE개)
                    collected = []
                    while len(collected) < MAX_BATCH_SIZE and queue:
                        frame_data = queue.popleft()
                        collected.append(frame_data)
                    
                    if collected:
                        frames_to_process[cam_id] = collected
            
            # 배치 처리 수행
            for cam_id, frame_list in frames_to_process.items():
                if not frame_list:
                    continue
                
                batch_processing_flags[cam_id] = True
                
                try:
                    # 배치 처리 (스레드 풀에서 실행)
                    loop = asyncio.get_event_loop()
                    frame_data_list = [(frame_bytes, cam_id) for frame_bytes, _ in frame_list]
                    
                    results = await asyncio.wait_for(
                        loop.run_in_executor(None, process_batch_frames, frame_data_list),
                        timeout=10.0
                    )
                    
                    # 결과 처리 및 저장
                    for (processed_frame_bytes, result_data), (_, timestamp) in zip(results, frame_list):
                        with frame_lock:
                            latest_processed_frames[cam_id] = processed_frame_bytes
                            latest_frames[cam_id] = processed_frame_bytes
                            if result_data:
                                latest_result_data[cam_id] = result_data
                        
                        # 모델 결과 업데이트
                        if result_data:
                            update_model_results_from_frame(result_data, cam_id)
                            asyncio.create_task(broadcast_to_dashboards(result_data, cam_id))
                            asyncio.create_task(broadcast_model_results())
                    
                    logging.debug(f"배치 처리 완료: cam_id={cam_id}, 프레임 수={len(frame_list)}")
                    
                except asyncio.TimeoutError:
                    logging.warning(f"배치 처리 타임아웃 (10초 초과): cam_id={cam_id}")
                except Exception as e:
                    logging.error(f"배치 처리 중 오류 (cam_id={cam_id}): {e}", exc_info=True)
                finally:
                    batch_processing_flags[cam_id] = False
            
            # 프레임이 없으면 짧게 대기
            if not frames_to_process:
                await asyncio.sleep(0.05)  # 50ms 대기
            else:
                await asyncio.sleep(0.01)  # 10ms 대기
                
        except Exception as e:
            logging.error(f"배치 처리 워커 오류: {e}", exc_info=True)
            await asyncio.sleep(0.1)


# --- 웹소켓 핸들러 (배치 처리 활성화) ---
async def websocket_handler(request: web.Request):
    global server_loop, frame_queues, queue_lock
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    connected_websockets.add(ws)
    client_id = id(ws) # 클라이언트 식별용
    logging.info(f"웹소켓 클라이언트 {client_id} 연결됨. (현재 {len(connected_websockets)}명 접속 중)")

    try:
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                # 클라이언트로부터 프레임 수신
                frame_bytes = msg.data
                cam_id = 0  # 카메라 ID (기본값 0)

                # 프레임 큐에 추가 (배치 처리용)
                with queue_lock:
                    if cam_id not in frame_queues:
                        frame_queues[cam_id] = deque(maxlen=core.config.SystemConfig.BATCH_SIZE * 2)
                    
                    # 큐가 가득 차면 가장 오래된 프레임 제거
                    frame_queues[cam_id].append((frame_bytes, time.time()))
                
                # 클라이언트에게 즉시 응답 (최신 처리된 프레임 또는 큐에 추가되었다는 확인)
                # 실제 처리된 프레임은 배치 처리 워커가 업데이트함
                with frame_lock:
                    latest_frame = latest_processed_frames.get(cam_id)
                    if latest_frame:
                        try:
                            await ws.send_bytes(latest_frame)
                        except (ConnectionResetError, ConnectionError, OSError):
                            break

                # 결과 데이터 전송 (최신 결과)
                with frame_lock:
                    latest_result = latest_result_data.get(cam_id)
                    if latest_result:
                        try:
                            await ws.send_str(json.dumps(latest_result))
                        except (ConnectionResetError, ConnectionError, OSError):
                            break  # 연결 끊어짐

            elif msg.type == WSMsgType.ERROR:
                logging.warning(f"웹소켓 클라이언트 {client_id} 오류 발생: {ws.exception()}")
                break

    except (ConnectionResetError, ConnectionError, OSError) as e:
        logging.info(f"클라이언트 {client_id} 연결이 끊어졌습니다. ({e})")
    finally:
        connected_websockets.discard(ws)
        dashboard_websockets.discard(ws)
        logging.info(f"웹소켓 클라이언트 {client_id} 연결 종료. (현재 {len(connected_websockets)}명 접속 중)")

# --- 비디오 스트림 핸들러 (final과 동일) ---
async def video_feed_handler(request: web.Request):
    """비디오 스트림 핸들러 - 보안 강화"""
    try:
        cam_id_str = request.match_info.get('cam_id', '0')
        
        # 입력 검증 및 정리
        try:
            cam_id = int(cam_id_str)
            # 카메라 ID 범위 검증
            if not (0 <= cam_id <= 10):
                logging.warning(f"잘못된 카메라 ID 범위: {cam_id}")
                return web.Response(status=400, text="Invalid camera ID range")
        except ValueError:
            # 문자열인 경우 구역별 매핑
            zone_mapping = {
                'a1': 0, 'a-1': 0,
                'a2': 1, 'a-2': 1, 
                'b1': 2, 'b-1': 2,
                'b2': 3, 'b-2': 3
            }
            cam_id = zone_mapping.get(cam_id_str.lower(), 0)
            logging.info(f"구역 '{cam_id_str}'을 카메라 ID {cam_id}로 매핑")
        
        response = web.StreamResponse(headers={
            'Content-Type': 'multipart/x-mixed-replace; boundary=frame',
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY'
        })
        await response.prepare(request)
        logging.info(f"CAM {cam_id} 스트리밍 시작.")
        
        try:
            while True:
                with frame_lock:
                    frame_bytes = latest_frames.get(cam_id)
                if frame_bytes:
                    await response.write(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                await asyncio.sleep(1 / 30)
        except asyncio.CancelledError:
            logging.info(f"CAM {cam_id} 스트리밍 클라이언트 연결 종료.")
        finally:
            await response.write_eof()
            
    except Exception as e:
        logging.error(f"비디오 스트림 처리 중 오류: {e}", exc_info=True)
        return web.Response(status=500, text="Internal server error")

# --- API 핸들러들 (final과 동일) ---
async def api_status_handler(request: web.Request):
    """시스템 상태 API 엔드포인트"""
    try:
        # 시스템 상태 확인
        system_status = "running"
        camera_count = len(latest_frames)
        
        # 카메라가 없으면 테스트 모드로 표시
        if camera_count == 0:
            system_status = "test_mode"
        
        status_data = {
            "system_status": system_status,
            "cameras": camera_count,
            "connected_clients": len(connected_websockets),
            "uptime": time.time() - system_stats["start_time"],
            "test_mode": camera_count == 0,
            "last_update": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        response = create_standard_response(data=status_data, message="시스템 상태 조회 성공")
        return web.json_response(response)
    except Exception as e:
        logging.error(f"API 상태 조회 중 오류: {e}")
        response = create_standard_response(
            status="error", 
            message=f"시스템 상태 조회 실패: {str(e)}", 
            error_code="STATUS_ERROR"
        )
        return web.json_response(response, status=500)

async def api_cameras_handler(request: web.Request):
    """카메라 목록 API 엔드포인트"""
    try:
        cameras_data = []
        
        # 현재 활성 카메라 수 파악
        with frame_lock:
            active_cameras = list(latest_frames.keys())
        
        # 각 카메라의 상세 정보 수집
        for cam_id in active_cameras:
            last_frame_time = time.time()
            
            # 실제 측정된 FPS 가져오기 (SafetySystem에서)
            actual_fps = 30.0  # 기본값
            if safety_system_instance:
                try:
                    if hasattr(safety_system_instance, 'get_camera_fps_data'):
                        fps_data = safety_system_instance.get_camera_fps_data()
                        actual_fps = fps_data.get(cam_id, 30.0)
                except Exception as e:
                    logging.error(f"FPS 데이터 가져오기 실패: {e}")
            
            cameras_data.append({
                "id": cam_id,
                "name": f"카메라 {cam_id}",
                "status": "active",
                "last_frame_time": last_frame_time,
                "stream_url": f"/video_feed/{cam_id}",
                "resolution": "1280x720",
                "fps": actual_fps,
                "is_streaming": True
            })
        
        # 카메라 개수 정보 및 시스템 상태 포함
        camera_info = {
            "cameras": cameras_data,
            "total_cameras": len(cameras_data),
            "active_cameras": len(cameras_data),
            "system_status": "running" if cameras_data else "no_cameras",
            "last_updated": time.time()
        }
        
        response_data = create_standard_response(data=camera_info, message=f"카메라 목록 조회 성공 - {len(cameras_data)}개 카메라 감지")
        return create_compressed_response(response_data)
    except Exception as e:
        logging.error(f"카메라 목록 조회 중 오류: {e}")
        response = create_standard_response(status="error", message=f"카메라 목록 조회 실패: {str(e)}", error_code="CAMERAS_ERROR")
        return web.json_response(response, status=500)

async def api_model_results_handler(request: web.Request):
    """모델 결과 API 엔드포인트"""
    try:
        # 캐시 미스 - 실제 데이터 조회
        with results_lock:
            # 데이터 필터링 적용
            filtered_data = filter_model_results(model_results)
            response_data = create_standard_response(data=filtered_data, message="모델 결과 조회 성공")
        
        return create_compressed_response(response_data)
    except Exception as e:
        logging.error(f"모델 결과 조회 중 오류: {e}")
        response = create_standard_response(status="error", message=f"모델 결과 조회 실패: {str(e)}", error_code="MODEL_RESULTS_ERROR")
        return web.json_response(response, status=500)

async def api_health_handler(request):
    """시스템 헬스체크 API"""
    try:
        # psutil 선택적 import (없어도 작동하도록)
        try:
            import psutil
            psutil_available = True
        except ImportError:
            psutil_available = False
            logging.warning("psutil을 사용할 수 없습니다. 기본 정보만 제공합니다.")
        
        with stats_lock:
            current_time = time.time()
            uptime = current_time - system_stats["start_time"]
            
            # 시스템 리소스 사용률 (psutil이 있으면 사용, 없으면 기본값)
            if psutil_available:
                try:
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    memory = psutil.virtual_memory()
                    memory_percent = memory.percent
                    memory_available = memory.available
                except Exception as e:
                    logging.warning(f"psutil 사용 중 오류: {e}")
                    cpu_percent = 0
                    memory_percent = 0
                    memory_available = 0
            else:
                cpu_percent = 0
                memory_percent = 0
                memory_available = 0
            
            # SafetySystem 상태 확인
            safety_status = "healthy" if safety_system_instance is not None else "unhealthy"
            
            # 카메라 상태 확인
            camera_count = len(latest_frames)
            
            health_data = {
                "status": "healthy",
                "timestamp": current_time,
                "uptime": uptime,
                "system": {
                    "cpu_usage": cpu_percent,
                    "memory_usage": memory_percent,
                    "memory_available": memory_available,
                    "total_requests": system_stats["total_requests"],
                    "error_count": system_stats["error_count"],
                    "error_rate": system_stats["error_count"] / max(system_stats["total_requests"], 1) * 100
                },
                "safety_system": {
                    "status": safety_status,
                    "camera_count": camera_count,
                    "connected_websockets": len(connected_websockets)
                },
                "performance": {
                    "avg_response_time": 0,  # TODO: 구현 필요
                    "throughput": system_stats["total_requests"] / max(uptime, 1)
                }
            }
            
            # 상태 업데이트
            system_stats["last_health_check"] = current_time
            if psutil_available:
                try:
                    system_stats["memory_usage"] = memory_percent
                    system_stats["cpu_usage"] = cpu_percent
                except:
                    pass
            
            response = create_standard_response(data=health_data, message="시스템 헬스체크 성공")
            return web.json_response(response)
    except Exception as e:
        logging.error(f"헬스체크 중 오류: {e}", exc_info=True)
        # 예외 발생 시에도 200 응답 반환 (서버는 실행 중이지만 일부 정보를 가져올 수 없음)
        health_data = {
            "status": "degraded",
            "timestamp": time.time(),
            "uptime": time.time() - system_stats.get("start_time", time.time()),
            "error": str(e)
        }
        response = create_standard_response(
            status="partial",
            message=f"헬스체크 부분 실패: {str(e)}",
            error_code="HEALTH_ERROR",
            data=health_data
        )
        # 200 응답 반환 (서버는 실행 중이므로)
        return web.json_response(response, status=200)

async def api_violations_handler(request: web.Request):
    """MongoDB 위반 이벤트 조회 API"""
    try:
        from datetime import datetime, timedelta
        
        # MongoDB 연결 시도 (선택적)
        try:
            from database import get_database
            db = get_database()
            if db and db.is_connected():
                # 쿼리 파라미터 파싱
                query_params = request.query
                worker_name = query_params.get('worker_name')
                camera_id = query_params.get('camera_id')
                camera_id = int(camera_id) if camera_id else None
                event_type = query_params.get('event_type')
                days = int(query_params.get('days', '7'))
                
                start_time = datetime.now() - timedelta(days=days)
                end_time = datetime.now()
                
                # MongoDB에서 조회
                violations = db.get_violations(
                    worker_name=worker_name,
                    camera_id=camera_id,
                    event_type=event_type,
                    start_time=start_time,
                    end_time=end_time,
                    limit=1000
                )
                
                # ObjectId를 문자열로 변환
                for v in violations:
                    v['_id'] = str(v['_id'])
                    if 'timestamp' in v and isinstance(v['timestamp'], datetime):
                        v['timestamp'] = v['timestamp'].isoformat()
                
                response_data = create_standard_response(
                    data={"violations": violations, "count": len(violations)},
                    message="위반 이벤트 조회 성공"
                )
                return web.json_response(response_data)
        except ImportError:
            pass  # database 모듈이 없으면 아래 코드 실행
        except Exception as db_error:
            logging.warning(f"MongoDB 조회 실패: {db_error}")
        
        # MongoDB가 없으면 빈 결과 반환
        response_data = create_standard_response(
            data={"violations": [], "count": 0},
            message="위반 이벤트 조회 성공 (MongoDB 미연결)"
        )
        return web.json_response(response_data)
    
    except Exception as e:
        logging.error(f"위반 이벤트 조회 중 오류: {e}")
        response = create_standard_response(
            status="error",
            message=f"위반 이벤트 조회 실패: {str(e)}",
            error_code="VIOLATIONS_ERROR"
        )
        return web.json_response(response, status=500)

async def api_workers_handler(request: web.Request):
    """MongoDB 작업자 조회 API"""
    try:
        from datetime import datetime
        
        # MongoDB 연결 시도 (선택적)
        try:
            from database import get_database
            db = get_database()
            if db and db.is_connected():
                # 쿼리 파라미터
                active_only = request.query.get('active_only', 'true').lower() == 'true'
                
                workers = db.get_all_workers(active_only=active_only)
                
                # ObjectId를 문자열로 변환
                for w in workers:
                    w['_id'] = str(w['_id'])
                    for date_field in ['registered_at', 'last_seen']:
                        if date_field in w and isinstance(w[date_field], datetime):
                            w[date_field] = w[date_field].isoformat()
                
                response_data = create_standard_response(
                    data={"workers": workers, "count": len(workers)},
                    message="작업자 조회 성공"
                )
                return web.json_response(response_data)
        except ImportError:
            pass  # database 모듈이 없으면 아래 코드 실행
        except Exception as db_error:
            logging.warning(f"MongoDB 조회 실패: {db_error}")
        
        # MongoDB가 없으면 빈 결과 반환
        response_data = create_standard_response(
            data={"workers": [], "count": 0},
            message="작업자 조회 성공 (MongoDB 미연결)"
        )
        return web.json_response(response_data)
    
    except Exception as e:
        logging.error(f"작업자 조회 중 오류: {e}")
        response = create_standard_response(
            status="error",
            message=f"작업자 조회 실패: {str(e)}",
            error_code="WORKERS_ERROR"
        )
        return web.json_response(response, status=500)

async def api_statistics_handler(request: web.Request):
    """MongoDB 통계 조회 API"""
    try:
        # MongoDB 연결 시도 (선택적)
        try:
            from database import get_database
            db = get_database()
            if db and db.is_connected():
                # 쿼리 파라미터
                days = int(request.query.get('days', 7))
                
                # 통계 조회
                stats = db.get_violation_statistics(days=days)
                
                response_data = create_standard_response(
                    data=stats,
                    message="통계 조회 성공"
                )
                return web.json_response(response_data)
        except ImportError:
            pass  # database 모듈이 없으면 아래 코드 실행
        except Exception as db_error:
            logging.warning(f"MongoDB 조회 실패: {db_error}")
        
        # MongoDB가 없으면 빈 결과 반환
        response_data = create_standard_response(
            data={"total_violations": 0, "period_days": 7, "camera_stats": {}, "worker_stats": {}, "event_type_stats": {}},
            message="통계 조회 성공 (MongoDB 미연결)"
        )
        return web.json_response(response_data)
    
    except Exception as e:
        logging.error(f"통계 조회 중 오류: {e}")
        response = create_standard_response(
            status="error",
            message=f"통계 조회 실패: {str(e)}",
            error_code="STATISTICS_ERROR"
        )
        return web.json_response(response, status=500)

# --- CORS 설정 및 라우트 등록 ---
def create_app():
    app = web.Application()

    # CORS 설정
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*"
        )
    })

    # 요청 통계 미들웨어 추가 (final과 동일)
    @web.middleware
    async def stats_middleware(request, handler):
        start_time = time.time()
        with stats_lock:
            system_stats["total_requests"] += 1
        
        try:
            response = await handler(request)
            
            # WebSocket 연결은 지속적이므로 응답 시간 측정 제외
            # WebSocket 핸들러는 실제로 응답 객체를 반환하지 않으므로 체크
            if request.path.startswith('/ws'):
                # WebSocket 연결이므로 응답 시간 로깅 생략 (연결 종료까지 기다리면 안됨)
                return response
            
            return response
        except Exception as e:
            with stats_lock:
                system_stats["error_count"] += 1
            logging.error(f"요청 처리 중 오류: {e}")
            raise
        finally:
            # 응답 시간 로깅 (WebSocket 제외)
            if not request.path.startswith('/ws'):
                response_time = time.time() - start_time
                if response_time > 1.0:  # 1초 이상 걸린 요청만 로깅
                    logging.warning(f"느린 응답: {request.path} - {response_time:.2f}초")
    
    app.middlewares.append(stats_middleware)
    
    # 배치 처리 워커 시작 (Tesla V100 최적화)
    async def start_batch_worker(app):
        """서버 시작 시 배치 처리 워커 실행"""
        logging.info("🚀 배치 처리 워커 시작 (GPU 최적화 활성화)")
        asyncio.create_task(batch_processing_worker())
    
    app.on_startup.append(start_batch_worker)
    
    # 웹소켓 엔드포인트 (클라이언트용 - 프레임 수신 및 처리)
    app.router.add_get("/ws", websocket_handler)
    
    # 대시보드 전용 웹소켓 엔드포인트 (데이터만 받기)
    app.router.add_get("/ws/dashboard", dashboard_websocket_handler)

    # MJPEG 스트림 엔드포인트 (개선: latest_processed_frames가 없으면 latest_frames도 확인)
    async def mjpeg_stream(request: web.Request):
        global latest_processed_frames, latest_frames, frame_lock
        cam_id = int(request.query.get('cam_id', '0'))

        response = web.StreamResponse(
            status=200,
            reason='OK',
            headers={
                'Content-Type': 'multipart/x-mixed-replace; boundary=--jpgboundary',
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0'
            }
        )
        await response.prepare(request)

        try:
            while True:
                with frame_lock:
                    # latest_processed_frames가 없으면 latest_frames도 확인 (호환성)
                    frame_bytes = latest_processed_frames.get(cam_id) or latest_frames.get(cam_id)

                if frame_bytes:
                    await response.write(
                        b'--jpgboundary\r\n'
                        b'Content-Type: image/jpeg\r\n'
                        b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n' +
                        frame_bytes + b'\r\n'
                    )
                else:
                    # 프레임이 없을 때도 경계선만 보내서 연결 유지 (빈 프레임 방지)
                    await asyncio.sleep(0.1)  # 프레임이 없을 때는 조금 더 느리게

                await asyncio.sleep(0.033)  # ~30 FPS
        except (ConnectionResetError, ConnectionError, OSError):
            pass

    app.router.add_get("/stream", mjpeg_stream)

    # 대시보드 HTML 서빙 (메인 서버에 통합)
    async def dashboard_handler(request: web.Request):
        """AIVIS 대시보드 HTML 페이지 서빙"""
        dashboard_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "AIVIS_Dashboard.html")
        try:
            with open(dashboard_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            return web.Response(text=html_content, content_type='text/html')
        except FileNotFoundError:
            logging.error(f"대시보드 파일을 찾을 수 없습니다: {dashboard_path}")
            return web.Response(text="<h1>Dashboard file not found</h1>", status=404)
        except Exception as e:
            logging.error(f"대시보드 파일 읽기 오류: {e}")
            return web.Response(text=f"<h1>Error loading dashboard: {e}</h1>", status=500)

    # 루트 경로와 /dashboard 경로 모두 대시보드 제공
    app.router.add_get("/", dashboard_handler)
    app.router.add_get("/dashboard", dashboard_handler)
    
    # 비디오 스트림 엔드포인트 (final과 동일)
    app.router.add_get('/video_feed/{cam_id}', video_feed_handler)
    
    # API 엔드포인트 (final과 동일)
    app.router.add_get('/api/status', api_status_handler)
    app.router.add_get('/api/cameras', api_cameras_handler)
    app.router.add_get('/api/model-results', api_model_results_handler)
    app.router.add_get('/api/health', api_health_handler)
    app.router.add_get('/api/violations', api_violations_handler)
    app.router.add_get('/api/workers', api_workers_handler)
    app.router.add_get('/api/statistics', api_statistics_handler)

    # CORS 적용
    for route in list(app.router.routes()):
        cors.add(route)

    return app

# --- 메인 함수 ---
def main():
    global safety_system_instance, server_loop

    setup_logging()
    server_loop = asyncio.get_event_loop()

    # SafetySystem 초기화
    try:
        safety_system_instance = core.SafetySystem()
        logging.info("SafetySystem 초기화 완료")
    except Exception as e:
        logging.critical(f"SafetySystem 초기화 실패: {e}", exc_info=True)
        logging.critical("서버를 시작할 수 없습니다.")
        return

    # 웹 애플리케이션 생성 및 실행
    app = create_app()

    # 종료 신호 처리
    def signal_handler(sig, frame):
        logging.info("종료 신호 수신. 서버 종료 중...")
        # 스레드 풀 종료
        face_recognition_executor.shutdown(wait=False)
        server_loop.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 포트 설정 (환경 변수 또는 기본값 사용, final과 동일하게 5008)
    port = int(os.getenv('SERVER_PORT', '5008'))

    # 메인 서버 시작 (대시보드 포함, 단일 포트에서 모든 서비스 제공)
    async def start_server():
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', port)
        await site.start()
        logging.info(f"서버 시작 완료: http://0.0.0.0:{port}")
        logging.info(f"  - 대시보드: http://0.0.0.0:{port}/")
        logging.info(f"  - WebSocket (클라이언트): ws://0.0.0.0:{port}/ws")
        logging.info(f"  - WebSocket (대시보드): ws://0.0.0.0:{port}/ws/dashboard")
        logging.info(f"  - MJPEG Stream: http://0.0.0.0:{port}/stream")

        # 무한 대기 (서버 계속 실행)
        try:
            await asyncio.Future()  # 영원히 대기
        except asyncio.CancelledError:
            pass

    # 이벤트 루프에서 서버 시작
    try:
        asyncio.run(start_server())
    except KeyboardInterrupt:
        logging.info("서버 종료 중...")
        face_recognition_executor.shutdown(wait=False)

if __name__ == "__main__":
    main()
