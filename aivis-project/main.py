# main.py (서버 실행에 사용되지 않음)
from fastapi import FastAPI
import uvicorn

# 1. FastAPI 앱 생성
app = FastAPI()

# 2. 기본 접속 주소 ("/")로 GET 요청이 오면
@app.get("/")
def read_root():
    # 이 메시지를 반환
    return {"message": "AIVIS AI 서버가 성공적으로 실행되었습니다!"}

# (나중에 여기에 영상 프레임을 받는 @app.post("/analyze_frame") 코드를 추가)

# 3. 이 파일을 직접 실행했을 때
if __name__ == "__main__":
    # Uvicorn 서버를 0.0.0.0 호스트의 8000 포트로 실행
    # 0.0.0.0은 Docker/KubeSphere 환경에서 필수입니다.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)