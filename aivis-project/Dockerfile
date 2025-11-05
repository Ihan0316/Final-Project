# 기본 이미지: Ubuntu 22.04
FROM ubuntu:22.04

# 작업 디렉토리 설정
WORKDIR /app

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 시스템 패키지 및 Python 설치
RUN sed -i 's/archive.ubuntu.com/kr.archive.ubuntu.com/g' /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        python3-dev \
        fonts-nanum* \
        libgl1-mesa-glx \
        libglib2.0-0 \
        wget \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgstreamer1.0-0 \
        libgstreamer-plugins-base1.0-0 \
        build-essential \
        g++ \
        gcc \
    && ln -s /usr/bin/python3 /usr/bin/python

# Python 의존성 설치 (캐시 최적화를 위해 먼저 복사)
COPY requirements.txt .

# Python 라이브러리 설치 및 OpenCV 최적화
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless cv2 2>/dev/null || true \
    && pip install --no-cache-dir opencv-contrib-python-headless==4.7.0.72 \
    && rm -rf ~/.cache/pip/* /tmp/* /var/tmp/* \
    && apt-get purge -y build-essential g++ gcc \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# InsightFace 패치 - blobFromImages 문제 해결
RUN python3 << 'EOF'
import os, glob

# InsightFace 경로 찾기
insightface_dirs = glob.glob('/usr/local/lib/python3.*/dist-packages/insightface')
if insightface_dirs:
    insightface_dir = insightface_dirs[0]
    arcface_file = os.path.join(insightface_dir, 'model_zoo', 'arcface_onnx.py')

    if os.path.exists(arcface_file):
        try:
            with open(arcface_file, 'r') as f:
                content = f.read()

            modified = False

            # blobFromImages 호출 찾기 및 교체 (간단한 방식)
            if 'blob = cv2.dnn.blobFromImages(' in content:
                # 정규식을 사용하여 안전하게 교체
                import re
                # 패턴: blob = cv2.dnn.blobFromImages(...);
                # 임의의 매개변수까지 포함하여 한 줄로 교체
                pattern = r'blob\s*=\s*cv2\.dnn\.blobFromImages\s*\([^)]+\)'
                new_code = 'blob = self._create_blob_batch(imgs, input_size)'

                if re.search(pattern, content):
                    # 각 라인의 들여쓰기를 유지하면서 교체
                    lines = content.split('\n')
                    new_lines = []
                    for line in lines:
                        if re.search(pattern, line):
                            indent = len(line) - len(line.lstrip())
                            new_lines.append(' ' * indent + new_code)
                            modified = True
                        else:
                            new_lines.append(line)
                    content = '\n'.join(new_lines)

            # _create_blob_batch 메서드가 없으면 추가
            if '_create_blob_batch' not in content and modified:
                # get_feat 메서드 뒤에 클래스 메서드로 추가
                get_feat_pos = content.find('def get_feat(self, imgs):')
                if get_feat_pos >= 0:
                    # 다음 메서드 또는 클래스 찾기
                    insert_pos = content.find('\nclass ', get_feat_pos)
                    if insert_pos == -1:
                        insert_pos = content.find('\n    def ', get_feat_pos)
                    if insert_pos == -1:
                        # get_feat 메서드의 끝 찾기 (다음 def나 class가 없으면 문서 끝)
                        insert_pos = len(content)

                    # 메서드 추가
                    new_method = '''
    def _create_blob_batch(self, imgs, input_size):
        """Custom batch blob creation to avoid OpenCV issues"""
        import cv2
        import numpy as np
        try:
            return cv2.dnn.blobFromImages(imgs, 1.0/self.input_std, input_size, (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        except (AttributeError, TypeError):
            return np.array([cv2.resize(img, input_size).astype('float32').transpose(2,0,1).reshape(1,3,*input_size[::-1]) for img in imgs])

'''
                    content = content[:insert_pos] + new_method + content[insert_pos:]
                    modified = True

            # 파일 저장
            if modified:
                with open(arcface_file, 'w') as f:
                    f.write(content)
                print("✓ Successfully patched arcface_onnx.py")
            else:
                print("⚠ No changes needed or already patched")

        except Exception as e:
            print(f"✗ Patch error: {e}")
            import traceback
            traceback.print_exc()
else:
    print("⚠ InsightFace not found")
EOF

# 애플리케이션 코드 복사 (.dockerignore에 명시된 파일/폴더는 제외)
COPY . .

# 작업 디렉토리를 /app/src로 변경
WORKDIR /app/src

# 포트 노출 (Kubernetes/KubeSphere 배포용)
EXPOSE 8000

# 헬스 체크 설정
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD wget --quiet --tries=1 --spider http://localhost:8000/ || exit 1

# 서버 시작 명령
CMD ["python", "server.py"]
