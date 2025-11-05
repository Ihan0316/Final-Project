# core.py - SafetySystem 클래스 (모델 로딩 및 관리)
import os
import cv2
import torch
import logging
import numpy as np
import platform
import faiss
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
from ultralytics.engine.results import Keypoints
from insightface.app import FaceAnalysis

import config
from utils import calculate_iou, clip_bbox_xyxy, is_person_horizontal, log_violation



class SafetySystem:
    def __init__(self):
        # 1. 장치 설정
        self.device_config = config.SystemConfig.get_device_config()
        self.device = self.device_config['device']
        logging.info(f"SafetySystem: 사용 장치 설정: {self.device.upper()}")

        # 2. 모델 로딩
        self.violation_model, self.pose_model = self._initialize_tracking_models()
        self.face_analyzer, self.face_database, self.super_res_model, self.super_res_net = self._initialize_face_recognition_models()

        if self.violation_model is None or self.pose_model is None:
            logging.error("필수 모델(Violation or Pose) 로딩에 실패했습니다.")
        else:
             logging.info("YOLO 모델 로딩 완료.")

        if self.face_analyzer is None or self.face_database is None:
            logging.warning("얼굴 인식 모델 또는 DB 로딩에 실패했습니다. 얼굴 인식 기능이 비활성화됩니다.")
        else:
            logging.info("얼굴 인식 모델 및 DB 로딩 완료.")

    def _initialize_tracking_models(self) -> Tuple[Optional[YOLO], Optional[YOLO]]:
        try:
            if not os.path.exists(config.Paths.YOLO_VIOLATION_MODEL):
                 raise FileNotFoundError(f"Violation 모델 파일 없음: {config.Paths.YOLO_VIOLATION_MODEL}")
            if not os.path.exists(config.Paths.YOLO_POSE_MODEL):
                 raise FileNotFoundError(f"Pose 모델 파일 없음: {config.Paths.YOLO_POSE_MODEL}")

            violation_model = YOLO(config.Paths.YOLO_VIOLATION_MODEL)
            pose_model = YOLO(config.Paths.YOLO_POSE_MODEL)

            # 모델을 올바른 장치로 이동
            violation_model.to(self.device)
            # 포즈 모델은 MPS에서 CPU로 자동 폴백될 수 있음 (server.py에서 처리)
            pose_model.to('cpu' if self.device == 'mps' else self.device)

            # CUDA 최적화 설정 (Tesla V100 최적화)
            if self.device == 'cuda':
                torch.backends.cudnn.benchmark = True  # cuDNN 자동 튜닝 활성화
                torch.backends.cudnn.deterministic = False  # 비결정적 알고리즘 허용 (속도 우선)
                # 메모리 할당자 최적화
                torch.cuda.set_per_process_memory_fraction(0.95)  # 95% 메모리 사용
                logging.info("CUDA 최적화 설정 완료 (cuDNN benchmark, 메모리 할당 최적화)")

            logging.info(f"Violation 모델 로드: {config.Paths.YOLO_VIOLATION_MODEL}")
            logging.info(f"Pose 모델 로드: {config.Paths.YOLO_POSE_MODEL}")

            return violation_model, pose_model
        except Exception as e:
            logging.error(f"YOLO 모델 초기화 실패: {e}", exc_info=True)
            return None, None

    def _initialize_face_recognition_models(self):
        face_analyzer = None
        face_database = None
        super_res_model = None

        try:
            # 1. InsightFace 분석기 로드
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
            face_analyzer = FaceAnalysis(
                name='buffalo_l',
                providers=providers,
                allowed_modules=['detection', 'recognition']
            )
            face_analyzer.prepare(ctx_id=0 if self.device == 'cuda' else -1)
            logging.info(f"InsightFace 모델 로드 완료 (Provider: {providers})")

            # 2. Faiss DB 로드
            face_database = self._load_face_database(config.Paths.FAISS_INDEX)

            # 3. Super Resolution 모델 로드 (선택 사항)
            super_res_model = None
            super_res_net = None
            if os.path.exists(config.Paths.EDSR_MODEL):
                try:
                    # DnnSuperResImpl 객체 생성 (호환성을 위해 유지)
                    if self.device == 'cuda' and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                        super_res_model = cv2.dnn_superres.DnnSuperResImpl_create()
                        super_res_model.readModel(config.Paths.EDSR_MODEL)
                        super_res_model.setModel("espcn", 4)
                        super_res_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                        super_res_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                        logging.info("EDSR (GPU) 초해상도 모델 로드 완료.")
                    else:
                        # CPU 또는 CUDA 없는 환경
                        super_res_model = cv2.dnn_superres.DnnSuperResImpl_create()
                        super_res_model.readModel(config.Paths.EDSR_MODEL)
                        super_res_model.setModel("espcn", 4)
                        logging.info("EDSR (CPU) 초해상도 모델 로드 완료.")

                    # DNN 네트워크 직접 로드 (getNet() 메서드 대체)
                    super_res_net = cv2.dnn.readNet(config.Paths.EDSR_MODEL)
                    if self.device == 'cuda' and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                        super_res_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                        super_res_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                    logging.info("초해상도 DNN 네트워크 직접 로드 완료.")
                except Exception as e_sr:
                    logging.warning(f"초해상도(SR) 모델 로드 실패: {e_sr}. (무시하고 진행)")
                    super_res_model = None
                    super_res_net = None
            else:
                 logging.warning(f"초해상도 모델 파일을 찾을 수 없습니다: {config.Paths.EDSR_MODEL}")

        except Exception as e:
            logging.error(f"얼굴 인식 시스템 초기화 실패: {e}", exc_info=True)
            face_analyzer = None # 실패 시 None으로 설정
            face_database = None
            super_res_model = None
            super_res_net = None

        return face_analyzer, face_database, super_res_model, super_res_net

    @staticmethod
    def _load_face_database(index_path: str) -> Optional[faiss.Index]:
        try:
            if not os.path.exists(index_path):
                index_path = os.path.normpath(os.path.join(config.BASE_DIR, "..", "face_index.faiss"))
                if not os.path.exists(index_path):
                    logging.warning(f"Faiss 인덱스 파일을 찾을 수 없습니다: {index_path}")
                    return None

            dimension = 512  # InsightFace 임베딩 차원
            index = faiss.read_index(index_path)

            logging.info(f"Faiss 인덱스 로드 완료. {index.ntotal}개 임베딩 포함.")
            return index
        except Exception as e:
            logging.error(f"Faiss 데이터베이스 로드 실패: {e}", exc_info=True)
            return None

    def cleanup(self):
        logging.info("SafetySystem 정리됨.")

    # --- 헬퍼 함수 (Static Methods) ---
    # 이 함수들은 server.py의 process_single_frame에서 호출됩니다.

    @staticmethod
    def _scale_boxes(boxes, w_scale, h_scale, names):
        scaled = defaultdict(list)
        if boxes is None or len(boxes) == 0:
            return scaled

        for box in boxes:
            try:
                class_name = names[int(box.cls[0])]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                scaled[class_name].append({
                    'bbox': (x1 * w_scale, y1 * h_scale, x2 * w_scale, y2 * h_scale),
                    'confidence': confidence
                })
            except Exception as e:
                logging.warning(f"박스 스케일링 중 오류: {e}")
        return scaled

    @staticmethod
    def _scale_poses(pose_result, w_scale, h_scale, orig_shape):
        scaled = []
        if pose_result.keypoints and pose_result.boxes is not None and pose_result.boxes.id is not None:
            tracker_ids = pose_result.boxes.id.int().cpu().numpy()
            for idx, (kpts, tracker_id) in enumerate(zip(pose_result.keypoints, tracker_ids)):
                try:
                    if torch.sum(kpts.conf > config.Thresholds.POSE_CONFIDENCE) >= config.Thresholds.MIN_VISIBLE_KEYPOINTS:
                        kpts_data = kpts.data.clone()
                        kpts_data[..., 0] *= w_scale
                        kpts_data[..., 1] *= h_scale
                        box = pose_result.boxes[idx].xyxy[0].cpu().numpy()
                        scaled_box = (box[0] * w_scale, box[1] * h_scale, box[2] * w_scale, box[3] * h_scale)

                        scaled.append({'keypoints': Keypoints(kpts_data, orig_shape), 'bbox_xyxy': scaled_box,
                                       'tracker_id': tracker_id})
                except Exception as e:
                    logging.warning(f"포즈 스케일링 중 오류: {e}")
        return scaled

    # ( ... _get_frame_batch, cleanup, _update_people_states, _draw_results, _create_grid_display 등등 ... 모두 삭제 ...)
    # ( ... _match_and_update_people, _check_fall_status, _check_safety_gear_status 등등 ... 모두 삭제 ...)
    # SafetySystem 클래스는 이제 모델 로딩과 헬퍼 함수(_scale_boxes, _scale_poses)만 제공합니다.
