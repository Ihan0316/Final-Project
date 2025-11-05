# Kubernetes 배포 가이드

## 📁 파일 구조

```
k8s/
├── deployment.yaml  # Deployment 설정
├── service.yaml     # Service 설정
└── README.md        # 이 파일
```

## 🚀 KubeSphere 웹 UI 배포 방법

### 1단계: Deployment 배포

1. **KubeSphere 콘솔 접속**

   - 브라우저에서 KubeSphere 웹 UI 접속

2. **Deployment 생성**

   - 왼쪽 메뉴: `Workloads` → `Deployments`
   - 네임스페이스: `estsoft-21` 선택
   - `Create` 버튼 클릭
   - `Create from YAML` 선택

3. **YAML 붙여넣기**
   - `deployment.yaml` 파일 내용 전체 복사
   - YAML 편집기에 붙여넣기
   - `Create` 클릭

### 2단계: Service 배포

1. **Service 생성**

   - 왼쪽 메뉴: `Network` → `Services`
   - 네임스페이스: `estsoft-21` 선택
   - `Create` 버튼 클릭
   - `Create from YAML` 선택

2. **YAML 붙여넣기**
   - `service.yaml` 파일 내용 전체 복사
   - YAML 편집기에 붙여넣기
   - `Create` 클릭

## ✅ 배포 확인

### Pod 상태 확인

1. `Workloads` → `Deployments` → `ai-model-server`
2. Pod 상태가 `Running`인지 확인
3. Pod 로그에서 다음 메시지 확인:
   ```
   서버 시작 완료: http://0.0.0.0:5008
   ```

### Service 확인

1. `Network` → `Services` → `ai-server-access`
2. 접속 주소 확인:
   - **NodePort**: `30001`
   - **외부 접속**: `http://<Node_IP>:30001`

## 🔧 설정 항목

### Deployment 설정

- **이미지**: `ihan0316/aivis-server:latest`
- **포트**: `5008`
- **리소스**:
  - CPU: 4 cores (request) / 8 cores (limit)
  - Memory: 27Gi (request) / 54Gi (limit)
  - GPU: 1 (request & limit)

### Service 설정

- **타입**: `NodePort`
- **포트**: `80`
- **타겟 포트**: `5008`
- **NodePort**: `30001`

## 🔄 업데이트 방법

### 이미지 업데이트 후 재배포

1. **로컬에서 이미지 빌드 및 푸시**

   ```bash
   cd /Users/ihanjo/Documents/Final-Project/aivis-project
   ./redeploy.sh
   ```

2. **KubeSphere에서 Pod 재시작**
   - `Workloads` → `Deployments` → `ai-model-server`
   - `More Actions` → `Restart`

## 📝 주의사항

- **네임스페이스**: `estsoft-21` (변경 시 모든 YAML에서 수정 필요)
- **포트**: 서버는 포트 `5008`에서 실행됩니다
- **이미지**: `imagePullPolicy: Always`로 설정되어 있어 Pod 재시작 시 최신 이미지를 자동으로 가져옵니다

## 🐛 문제 해결

### Pod가 시작되지 않는 경우

1. Pod 로그 확인
2. 리소스 확인 (GPU, 메모리)
3. 이미지 Pull 확인

### 연결이 안 되는 경우

1. Service의 `targetPort`가 `5008`인지 확인
2. Deployment의 `containerPort`가 `5008`인지 확인
3. Pod 상태가 `Running`인지 확인
