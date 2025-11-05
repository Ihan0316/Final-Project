# core.py
import logging
import time
import faiss
import os
from collections import defaultdict, deque
from threading import Thread
from queue import Queue, Empty, Full
from typing import List, Dict, Tuple, Optional

import psutil
import platform

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Keypoints
from insightface.app import FaceAnalysis

import config
import utils
from utils import (
    calculate_iou, clip_bbox_xyxy,
    is_person_horizontal, log_violation
)


class CameraStream:
    # --- [수정] 카메라 원본 해상도를 설정하여 CPU 부하 감소 ---
    def __init__(self, src: int):
        self.src = src
        self.stream = cv2.VideoCapture(self.src)
        if self.stream.isOpened():
            # 선호하는 해상도 설정 (카메라가 지원하지 않으면 무시됨)
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            logging.info(f"카메라 {self.src} 해상도 설정 시도: 1280x720")

        self.grabbed, self.frame = self.stream.read()
        if not self.grabbed: logging.error(f"카메라 {self.src}를 열 수 없습니다.")
        self.stopped = False
        self.thread = Thread(target=self.update, args=(), daemon=True)

    # --- [수정 완료] ---

    def start(self):
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            if not self.stream.isOpened():
                time.sleep(2.0)
                self.stream.open(self.src)
                continue
            self.grabbed, self.frame = self.stream.read()
            if not self.grabbed: self.stream.release()

    def read(self) -> Optional[np.ndarray]:
        return self.frame if self.grabbed else None

    def stop(self):
        self.stopped = True
        if self.thread.is_alive(): self.thread.join()
        if self.stream.isOpened(): self.stream.release()


class Person:
    def __init__(self, person_id: int, bbox_xyxy: Tuple[int, int, int, int]):
        self.id = person_id
        self.bbox_xyxy = bbox_xyxy
        self.name: str = f"Person_{self.id}"
        self.recognized = False
        self.recognition_needed = True
        self.inactive_frames = 0
        self.keypoints: Optional[Keypoints] = None
        self.current_event = "CHECKING"
        self.safety_status: Dict[str, str] = {rule: "CHECKING" for rule in config.Constants.SAFETY_RULES_MAP.keys()}
        self.last_log_time = 0
        self.last_recognition_time = 0
        self.fall_start_time: Optional[float] = None
        self.embedding_buffer = deque(maxlen=config.SystemConfig.EMBEDDING_BUFFER_SIZE)
        self.recognition_pending = False


class SafetySystem:
    def __init__(self):
        self._set_process_priority()
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

        self.violation_model, self.pose_model = self._initialize_tracking_models()
        self.face_analyzer, self.face_database, self.super_res_model = self._initialize_face_recognition_models()

        self.camera_streams = self._initialize_cameras()
        self.people_data: Dict[int, Dict[int, Person]] = {cam_id: {} for cam_id in config.SystemConfig.CAMERA_INDICES}
        self.display_fps = 0.0

        self.recognition_queue = Queue(maxsize=10)
        self.result_queue = Queue()
        self.logging_queue = Queue()

        self.logging_thread_stop = False
        self.logging_thread = Thread(target=self._logging_worker, daemon=True)
        self.logging_thread.start()
        logging.info("로깅 Worker 스레드가 시작되었습니다.")

        self.recognition_thread_stop = False
        self.recognition_thread = Thread(target=self._recognition_worker, daemon=True)
        self.recognition_thread.start()
        logging.info("얼굴 인식 Worker 스레드가 시작되었습니다.")

    def _set_process_priority(self):
        p = psutil.Process()
        try:
            if platform.system() == "Windows":
                p.nice(psutil.HIGH_PRIORITY_CLASS)
            else:
                p.nice(-10)
        except psutil.AccessDenied:
            logging.warning("프로세스 우선순위 변경에 실패했습니다. (관리자 권한이 필요할 수 있습니다)")

    def _initialize_tracking_models(self) -> Tuple[YOLO, YOLO]:
        logging.info(f"사용 가능 장치 감지: {self.device.upper()}")
        try:
            violation_model = YOLO(config.Paths.YOLO_VIOLATION_MODEL).to(self.device)
            pose_model = YOLO(config.Paths.YOLO_POSE_MODEL).to(self.device)
            logging.info("YOLO 모델 로딩 완료.")
            return violation_model, pose_model
        except Exception as e:
            raise SystemExit(f"치명적 오류: 추적 모델 로딩 실패 - {e}")

    def _initialize_face_recognition_models(self):
        logging.info("InsightFace 모델 로딩 중...")
        providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider'] if self.device == 'mps' else \
            ['CUDAExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
        face_analyzer = FaceAnalysis(name="buffalo_l", providers=providers)
        face_analyzer.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)
        logging.info(f"InsightFace 모델 로딩 완료. Provider: {face_analyzer.models['detection'].session.get_providers()}")
        face_database = self._load_face_database()
        super_res_model = None
        try:
            logging.info("EDSR 초해상도 모델 로딩 중...")
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            sr.readModel(config.Paths.EDSR_MODEL)
            sr.setModel("edsr", 4)
            super_res_model = sr
            logging.info("EDSR 모델 로딩 완료.")
        except Exception as e:
            logging.error(f"EDSR 모델 로딩 실패: {e}. 초해상도 기능 없이 계속합니다.")
        return face_analyzer, face_database, super_res_model

    def _logging_worker(self):
        while not self.logging_thread_stop:
            try:
                log_args = self.logging_queue.get(timeout=1)
                log_violation(*log_args)
            except Empty:
                continue
            except Exception as e:
                logging.error(f"[Logger] 로그 기록 중 오류 발생: {e}")

    def _recognition_worker(self):
        p = psutil.Process()
        try:
            if platform.system() == "Windows":
                p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            else:
                p.nice(10)
        except psutil.AccessDenied:
            pass

        while not self.recognition_thread_stop:
            try:
                cam_id, person_id, person_roi = self.recognition_queue.get(timeout=1)
                faces = self.face_analyzer.get(person_roi)
                if not faces and self.super_res_model is not None:
                    h, w, _ = person_roi.shape
                    if w < config.Thresholds.UPSCALE_THRESHOLD or h < config.Thresholds.UPSCALE_THRESHOLD:
                        upscaled_roi = self.super_res_model.upsample(person_roi)
                        faces = self.face_analyzer.get(upscaled_roi)
                if faces:
                    best_face = \
                    sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)[0]
                    embedding = best_face.normed_embedding
                    self.result_queue.put((cam_id, person_id, embedding))
            except Empty:
                continue
            except Exception as e:
                logging.error(f"[Worker] 얼굴 인식 중 오류 발생: {e}")

    def _load_face_database(self) -> Tuple[faiss.Index, np.ndarray]:
        try:
            index_path, labels_path = config.Paths.FAISS_INDEX, config.Paths.FAISS_LABELS
            if not (os.path.exists(index_path) and os.path.exists(labels_path)):
                raise SystemExit(f"Faiss DB 파일 없음 ({index_path}, {labels_path})")
            return faiss.read_index(index_path), np.load(labels_path, allow_pickle=True)
        except Exception as e:
            raise SystemExit(f"치명적 오류: Faiss DB 로드 실패 - {e}")

    def _initialize_cameras(self) -> List[CameraStream]:
        streams = [CameraStream(src=i).start() for i in config.SystemConfig.CAMERA_INDICES]
        if not any(s.grabbed for s in streams):
            raise SystemExit("치명적 오류: 사용 가능한 카메라가 없습니다.")
        return streams

    def run(self):
        logging.info("--- 시스템 초기화 완료. 실시간 감시를 시작합니다. ---")
        frames, cam_ids = self._get_frame_batch()
        if frames:
            logging.info("첫 프레임 탐지를 통해 시스템을 웜업합니다...")
            self._run_detection_and_recognition(frames, cam_ids)
            logging.info("웜업 완료.")
        frame_count = 0
        fps_start_time, fps_cycle_count = time.time(), 0
        try:
            while True:
                frames, cam_ids = self._get_frame_batch()
                if not frames:
                    time.sleep(0.01)
                    continue
                for cam_id in cam_ids:
                    if cam_id in self.people_data:
                        for p in self.people_data[cam_id].values(): p.inactive_frames += 1
                if frame_count % config.SystemConfig.DETECTION_INTERVAL == 0:
                    self._run_detection_and_recognition(frames, cam_ids)
                self._process_recognition_results()
                self._update_people_states(frames, cam_ids)
                processed_frames = self._draw_results(frames, cam_ids)
                if not config.SystemConfig.HEADLESS:
                    self._show_display(processed_frames)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                fps_cycle_count += len(frames)
                if time.time() - fps_start_time >= 1.0:
                    self.display_fps = fps_cycle_count / (time.time() - fps_start_time)
                    fps_start_time, fps_cycle_count = time.time(), 0
                frame_count += 1
        finally:
            self.cleanup()

    def _show_display(self, processed_frames: List[np.ndarray]):
        """처리된 프레임들을 그리드 형태로 화면에 표시합니다."""
        grid = self._create_grid_display(processed_frames)
        cv2.imshow('Intelligent Multi-Camera Safety System', grid)

    def _process_recognition_results(self):
        try:
            while not self.result_queue.empty():
                cam_id, person_id, embedding = self.result_queue.get_nowait()
                if cam_id in self.people_data and person_id in self.people_data[cam_id]:
                    p = self.people_data[cam_id][person_id]
                    p.embedding_buffer.append(embedding)
                    p.recognition_pending = False
                    if len(p.embedding_buffer) >= config.SystemConfig.EMBEDDING_BUFFER_SIZE:
                        avg_embedding = np.mean(list(p.embedding_buffer), axis=0)
                        norm = np.linalg.norm(avg_embedding)
                        if norm < 1e-6: continue
                        normalized_avg_embedding = (avg_embedding / norm).astype('float32')
                        faiss_index, labels = self.face_database
                        found_name, similarity = utils.find_best_match_faiss(
                            normalized_avg_embedding, faiss_index, labels, config.Thresholds.SIMILARITY
                        )
                        p.last_recognition_time = time.time()
                        p.embedding_buffer.clear()
                        if found_name != "Unknown":
                            p.name = found_name
                            p.recognized = True
                        p.recognition_needed = False
        except Empty:
            pass

    @torch.no_grad()
    def _run_detection_and_recognition(self, frames: List[np.ndarray], cam_ids: List[int]):
        h, w = config.SystemConfig.MODEL_INPUT_HEIGHT, config.SystemConfig.MODEL_INPUT_WIDTH
        resized_frames = [cv2.resize(f, (w, h)) for f in frames]
        pose_results = self.pose_model.track(resized_frames, persist=True, verbose=False,
                                             conf=config.Thresholds.YOLO_CONFIDENCE)
        vio_results = self.violation_model(resized_frames, verbose=False, conf=config.Thresholds.YOLO_CONFIDENCE)
        for i, cam_id in enumerate(cam_ids):
            frame = frames[i]
            orig_h, orig_w, _ = frame.shape
            h_scale, w_scale = orig_h / h, orig_w / w
            all_detections = self._scale_boxes(vio_results[i].boxes, w_scale, h_scale, self.violation_model.names)
            detected_poses = self._scale_poses(pose_results[i], w_scale, h_scale, (orig_h, orig_w))
            self._match_and_update_people(self.people_data[cam_id], detected_poses, frame)
            self._request_face_recognition(self.people_data[cam_id], cam_id, frame)
            for p in self.people_data[cam_id].values():
                self._update_person_event(p, all_detections)

    def _request_face_recognition(self, people: Dict[int, Person], cam_id: int, frame: np.ndarray):
        targets = [p for p in people.values() if p.recognition_needed and not p.recognition_pending]
        for p in targets:
            x1, y1, x2, y2 = map(int, p.bbox_xyxy)
            if (x2 - x1) < 64 or (y2 - y1) < 64: continue
            try:
                person_roi = frame[y1:y2, x1:x2].copy()
                self.recognition_queue.put_nowait((cam_id, p.id, person_roi))
                p.recognition_pending = True
            except Full:
                logging.warning("[Main] 인식 요청 큐가 꽉 찼습니다. 일부 요청을 건너뜁니다.")
                break

    def cleanup(self):
        logging.info("정리 작업 시작...")
        for stream in self.camera_streams: stream.stop()
        self.recognition_thread_stop = True
        if self.recognition_thread.is_alive(): self.recognition_thread.join(timeout=2)
        self.logging_thread_stop = True
        if self.logging_thread.is_alive(): self.logging_thread.join(timeout=2)
        cv2.destroyAllWindows()
        logging.info("시스템 종료.")

    def _get_frame_batch(self, *args, **kwargs):
        frames, cam_ids = [], []
        for i, stream in enumerate(self.camera_streams):
            frame = stream.read()
            if frame is not None:
                resized = cv2.resize(frame, (config.SystemConfig.DISPLAY_WIDTH, config.SystemConfig.DISPLAY_HEIGHT))
                frames.append(resized)
                cam_ids.append(config.SystemConfig.CAMERA_INDICES[i])
        return frames, cam_ids

    def _scale_boxes(self, boxes, w_scale, h_scale, names):
        scaled = defaultdict(list)
        for box in boxes:
            class_name = names[int(box.cls[0])]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            scaled[class_name].append((x1 * w_scale, y1 * h_scale, x2 * w_scale, y2 * h_scale))
        return scaled

    def _scale_poses(self, pose_result, w_scale, h_scale, orig_shape):
        scaled = []
        if pose_result.keypoints and pose_result.boxes.id is not None:
            tracker_ids = pose_result.boxes.id.int().cpu().numpy()
            for idx, (kpts, tracker_id) in enumerate(zip(pose_result.keypoints, tracker_ids)):
                if torch.sum(kpts.conf > 0.5) >= config.Thresholds.MIN_VISIBLE_KEYPOINTS:
                    kpts_data = kpts.data.clone()
                    kpts_data[..., 0] *= w_scale
                    kpts_data[..., 1] *= h_scale
                    box = pose_result.boxes[idx].xyxy[0].cpu().numpy()
                    scaled_box = (box[0] * w_scale, box[1] * h_scale, box[2] * w_scale, box[3] * h_scale)
                    scaled.append({'keypoints': Keypoints(kpts_data, orig_shape), 'bbox_xyxy': scaled_box,
                                   'tracker_id': tracker_id})
        return scaled

    def _match_and_update_people(self, people, poses, frame):
        for pose in poses:
            tracker_id = int(pose['tracker_id'])
            bbox_xyxy = pose['bbox_xyxy']
            clipped = clip_bbox_xyxy(bbox_xyxy, frame.shape[1], frame.shape[0])
            if not clipped: continue
            if tracker_id in people:
                p = people[tracker_id]
                p.bbox_xyxy = clipped
                p.keypoints = pose['keypoints']
                p.inactive_frames = 0
            else:
                if len(people) < config.SystemConfig.MAX_PEOPLE_TO_TRACK:
                    new_person = Person(tracker_id, clipped)
                    new_person.keypoints = pose['keypoints']
                    people[tracker_id] = new_person

    def _check_fall_status(self, person: Person) -> bool:
        """넘어짐 상태를 확인하고, 넘어짐이 감지되면 True를 반환합니다."""
        if person.keypoints and is_person_horizontal(person.keypoints, person.bbox_xyxy):
            if person.fall_start_time is None:
                person.fall_start_time = time.time()
            elif (time.time() - person.fall_start_time) > config.Thresholds.FALL_TIME:
                person.current_event = "FALL"
                person.recognition_needed = True
                return True
        else:
            person.fall_start_time = None
        return False

    def _check_safety_gear_status(self, person: Person, all_detections: Dict):
        """안전 장비 착용 상태를 확인하고 person.safety_status를 업데이트합니다."""
        person.safety_status = {rule: "CHECKING" for rule in person.safety_status}
        for rule, classes in config.Constants.SAFETY_RULES_MAP.items():
            comp_cls, viol_cls = classes["compliance"], classes["violation"]
            is_compliance_detected = False
            if comp_cls in all_detections:
                if any(calculate_iou(person.bbox_xyxy, box) > config.Thresholds.IOU_VIOLATION for box in
                       all_detections[comp_cls]):
                    is_compliance_detected = True
            is_violation_detected = False
            if viol_cls in all_detections:
                if any(calculate_iou(person.bbox_xyxy, box) > config.Thresholds.IOU_VIOLATION for box in
                       all_detections[viol_cls]):
                    is_violation_detected = True

            if is_violation_detected:
                person.safety_status[rule] = "VIOLATION"
            elif is_compliance_detected:
                person.safety_status[rule] = "SAFE"

    def _determine_final_event(self, person: Person):
        """safety_status를 기반으로 최종 이벤트를 결정합니다."""
        is_any_violation = any(status == "VIOLATION" for status in person.safety_status.values())
        is_all_safe = all(status == "SAFE" for status in person.safety_status.values())

        if is_any_violation:
            person.current_event = "VIOLATION"
        elif is_all_safe:
            person.current_event = "SAFE"
        else:
            person.current_event = "CHECKING"

        if person.current_event != "SAFE" and not person.recognized:
            person.recognition_needed = True

    def _update_person_event(self, person, all_detections):
        # 1. 넘어짐 상태 업데이트 (가장 높은 우선순위)
        if self._check_fall_status(person):
            return  # 넘어짐이 감지되면 다른 상태는 확인하지 않음

        # 2. 안전장비 상태 업데이트
        self._check_safety_gear_status(person, all_detections)

        # 3. 최종 이벤트 결정
        self._determine_final_event(person)

    def _update_people_states(self, frames, cam_ids):
        now = time.time()
        for i, cam_id in enumerate(cam_ids):
            if cam_id not in self.people_data: continue
            active_people = {tid: p for tid, p in self.people_data[cam_id].items() if
                             p.inactive_frames <= config.SystemConfig.MAX_INACTIVE_FRAMES}
            if len(active_people) != len(self.people_data[cam_id]):
                inactive_ids = set(self.people_data[cam_id].keys()) - set(active_people.keys())
                logging.info(f"[CAM-{cam_id}] 추적 종료 (ID: {list(inactive_ids)})")
            self.people_data[cam_id] = active_people

            for person in active_people.values():
                if person.current_event not in ["SAFE", "CHECKING"] and (
                        now - person.last_log_time) > config.Policy.EVENT_LOG_COOLDOWN:
                    log_event_name = person.current_event
                    if person.current_event == "VIOLATION":
                        violations = [rule for rule, status in person.safety_status.items() if status == "VIOLATION"]
                        log_event_name = ", ".join(violations)
                    log_args = (frames[i].copy(), person.name, log_event_name, cam_id)
                    try:
                        self.logging_queue.put_nowait(log_args)
                    except Full:
                        logging.warning("[Main] 로깅 큐가 꽉 찼습니다. 로그를 건너뜁니다.")
                    person.last_log_time = now
                if not person.recognition_needed:
                    is_unknown = person.name.startswith("Person_")
                    unknown_cd = (now - person.last_recognition_time) > config.Policy.UNKNOWN_RECOGNITION_COOLDOWN
                    known_cd = (now - person.last_recognition_time) > config.Policy.KNOWN_RECOGNITION_COOLDOWN
                    if (is_unknown and unknown_cd) or (not is_unknown and known_cd): person.recognition_needed = True

    def _draw_results(self, frames: List[np.ndarray], cam_ids: List[int]) -> List[np.ndarray]:
        processed_frames = []
        for i, cam_id in enumerate(cam_ids):
            frame = frames[i].copy()
            renderer = utils.TextRenderer(frame.shape)
            if cam_id in self.people_data:
                for p in self.people_data[cam_id].values():
                    x1, y1, x2, y2 = map(int, p.bbox_xyxy)
                    color = (0, 255, 0)
                    if p.current_event == "FALL":
                        color = (0, 0, 255)
                    elif p.current_event == "CHECKING":
                        color = (255, 255, 255)
                    elif p.current_event == "VIOLATION":
                        color = (0, 165, 255)

                    text_to_display = p.current_event
                    if p.current_event not in ["SAFE", "FALL"]:
                        non_safe_items = [rule for rule, status in p.safety_status.items() if status != "SAFE"]
                        if non_safe_items: text_to_display = ", ".join(non_safe_items)
                    text = f"{p.name}: {text_to_display}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    renderer.add_text(text, (x1, y1), color)

            renderer.add_text(f"CAM: {cam_id} | FPS: {self.display_fps:.1f}", (10, 35), (0, 255, 255))
            processed_frames.append(renderer.render_on(frame))
        return processed_frames

    def _create_grid_display(self, frames):
        if not frames: return np.zeros((config.SystemConfig.DISPLAY_HEIGHT, config.SystemConfig.DISPLAY_WIDTH, 3),
                                       np.uint8)
        h, w, _ = frames[0].shape
        num = len(frames)
        cols = int(np.ceil(np.sqrt(num)))
        rows = int(np.ceil(num / cols))
        grid = np.zeros((h * rows, w * cols, 3), np.uint8)
        for i, frame in enumerate(frames):
            r, c = divmod(i, cols)
            grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = frame
        return grid
