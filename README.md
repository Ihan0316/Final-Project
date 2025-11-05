# AIVIS - AI Vision Intelligence System

AI 기반 안전 관리 시스템으로, 실시간 얼굴 인식, 안전장비(PPE) 착용 감지, 위험 행동 탐지 기능을 제공합니다.

## 주요 기능

- **실시간 얼굴 인식**: FAISS 기반 고속 얼굴 매칭
- **안전장비(PPE) 감지**: 안전모, 마스크, 안전조끼 착용 여부 자동 감지
- **위험 행동 탐지**: 넘어짐 감지 및 위험 상황 알림
- **실시간 대시보드**: 웹 기반 모니터링 대시보드
- **다중 카메라 지원**: 여러 카메라 동시 처리

## 기술 스택

- **Backend**: Python 3.10+, FastAPI, aiohttp
- **AI 모델**:
  - YOLOv8 (객체 탐지 및 포즈 추정)
  - InsightFace (얼굴 인식)
  - EDSR (이미지 업스케일링)
- **인프라**: Docker, Kubernetes (선택사항)
- **데이터베이스**: FAISS (벡터 검색)

## 프로젝트 구조

```
aivis-project/
├── src/
│   ├── server.py      # 메인 서버 (FastAPI/aiohttp)
│   ├── config.py      # 설정 파일
│   ├── core.py        # 핵심 로직 (SafetySystem)
│   └── utils.py       # 유틸리티 함수
├── client.py          # 클라이언트 테스트 스크립트
├── main.py            # 진입점
├── Dockerfile         # Docker 빌드 설정
├── requirements.txt   # Python 의존성
└── AIVIS_Dashboard.html  # 웹 대시보드
```

## 설치 및 실행

### 필수 요구사항

- Python 3.10 이상
- CUDA 지원 GPU (선택사항, CPU 모드도 가능)
- 카메라 또는 비디오 파일

### 1. 저장소 클론

```bash
git clone <repository-url>
cd Final-Project/aivis-project
```

### 2. 환경 변수 설정

`.env` 파일을 생성하고 필요한 환경 변수를 설정하세요:

```env
# 모델 경로 (선택사항)
MODEL_BASE_DIR=/path/to/models

# 카메라 설정
CAMERA_INDICES=0  # 또는 0,1,2 (다중 카메라)

# 서버 설정
SERVER_PORT=5008

# 임계값 설정 (선택사항)
SIMILARITY_THRESHOLD=0.5
YOLO_CONFIDENCE=0.25
POSE_CONFIDENCE=0.20
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

**주의**: 일부 패키지 (faiss-gpu, onnxruntime-gpu)는 GPU 환경에 따라 설치가 필요합니다.

### 4. 모델 파일 준비

다음 모델 파일들을 `model/` 폴더에 배치하세요:

- `best.pt` - YOLO 안전장비 탐지 모델
- `yolov8n-pose.pt` - YOLO 포즈 추정 모델
- `EDSR_x4.pb` - EDSR 업스케일링 모델

### 5. 얼굴 데이터베이스 구축

FAISS 인덱스를 생성하기 위해 얼굴 이미지를 등록해야 합니다. (별도 스크립트 필요)

### 6. 서버 실행

```bash
python main.py
```

또는

```bash
python -m src.server
```

서버가 시작되면 다음 URL에서 접근할 수 있습니다:

- 대시보드: `http://localhost:5008/`
- WebSocket (클라이언트): `ws://localhost:5008/ws`
- WebSocket (대시보드): `ws://localhost:5008/ws/dashboard`
- MJPEG 스트림: `http://localhost:5008/stream`

## Docker를 사용한 실행

### Docker 이미지 빌드

```bash
docker build -t aivis:latest .
```

### Docker 컨테이너 실행

```bash
docker run -d \
  --name aivis \
  -p 5008:5008 \
  --gpus all \
  -v /path/to/models:/app/model \
  -v /path/to/data:/app/data \
  aivis:latest
```

## API 엔드포인트

- `GET /api/status` - 서버 상태 확인
- `GET /api/cameras` - 활성 카메라 목록
- `GET /api/model-results` - 최신 모델 결과
- `GET /api/health` - 헬스 체크
- `GET /api/violations` - 위반 사항 목록
- `GET /api/workers` - 작업자 정보
- `GET /api/statistics` - 통계 정보

## 설정

주요 설정은 `src/config.py`에서 관리됩니다:

- **Thresholds**: 탐지 임계값 설정
- **Paths**: 모델 및 데이터 경로
- **SystemConfig**: 시스템 동작 설정

환경 변수를 통해 대부분의 설정을 오버라이드할 수 있습니다.

## 보안 주의사항

- `.env` 파일은 git에 포함되지 않습니다. 반드시 별도로 관리하세요.
- 민감한 정보(API 키, 비밀번호 등)는 환경 변수로 관리하세요.
- 프로덕션 환경에서는 HTTPS를 사용하세요.

## 라이선스

[라이선스 정보를 여기에 추가하세요]

## 기여

이슈나 풀 리퀘스트를 환영합니다!

## 문의

문제가 발생하거나 질문이 있으시면 이슈를 등록해주세요.
