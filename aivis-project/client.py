# client.py
import cv2
import asyncio
import websockets
import time
import json
import numpy as np
import logging

# --- 설정 ---
# KubeSphere 서버의 IP 주소와 서비스에서 설정한 NodePort (Service YAML의 nodePort 값)
# ❗️❗️❗️ 서버 IP 주소를 정확히 확인하고 입력하세요 ❗️❗️❗️
SERVER_URI = "ws://210.125.70.71:30001/ws"
# 로컬 웹캠 인덱스 (보통 0)
CAMERA_INDEX = 0
# 초당 서버로 보낼 프레임 수 제한 (너무 높으면 서버 부하 증가)
# 서버 처리 속도를 고려하여 낮춤 (5-10 FPS 권장)
# 끊김 현상 방지를 위해 8로 낮춤
FPS_LIMIT = 8
# --- 설정 끝 ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def run_client():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        logging.error(f"오류: 카메라 {CAMERA_INDEX}를 열 수 없습니다.")
        return

    # 카메라 해상도 설정 시도 (낮춰서 전송량 줄이기)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logging.info(f"카메라 {CAMERA_INDEX} 열기 성공 ({width}x{height})")


    processed_frame = None # 서버로부터 받은 처리된 프레임 저장용
    last_results = {}     # 서버로부터 받은 마지막 JSON 결과 저장용
    frame_send_interval = 1.0 / FPS_LIMIT
    last_send_time = 0
    connection_status = "연결 시도 중..." # 연결 상태 표시용

    while True:
        try:
            # 연결 옵션 개선 (끊김 현상 방지)
            async with websockets.connect(
                SERVER_URI, 
                ping_interval=10,  # 20 → 10초 (더 자주 연결 확인)
                ping_timeout=5,    # ping 타임아웃
                open_timeout=30,
                close_timeout=10,
                max_size=10 * 1024 * 1024  # 10MB 최대 메시지 크기
            ) as websocket:
                logging.info(f"서버 {SERVER_URI}에 연결되었습니다.")
                connection_status = "연결됨"

                async def send_frames():
                    nonlocal last_send_time
                    while True:
                        current_time = time.time()
                        ret, frame = cap.read()
                        if not ret:
                            logging.warning("카메라에서 프레임을 읽는데 실패했습니다.")
                            await asyncio.sleep(0.1)
                            continue

                        # FPS 제한
                        if current_time - last_send_time < frame_send_interval:
                            await asyncio.sleep(0.01) # CPU 사용량 줄이기
                            continue

                        # 프레임을 JPEG로 인코딩 (압축률 조정 가능 - 65로 설정, 더 작은 파일 크기)
                        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
                        if not ret:
                            logging.warning("프레임 인코딩 실패")
                            continue

                        try:
                            # 바이너리 형태로 웹소켓 전송
                            await websocket.send(buffer.tobytes())
                            last_send_time = current_time
                            # logging.debug(f"프레임 전송 완료 (크기: {len(buffer)} bytes)")
                        except websockets.ConnectionClosed:
                            logging.warning("프레임 전송 중 연결이 끊겼습니다.")
                            break
                        except Exception as e:
                            logging.error(f"프레임 전송 중 오류: {e}")
                            await asyncio.sleep(1) # 오류 발생 시 잠시 대기

                        # asyncio가 다른 작업을 처리할 수 있도록 제어권 양보
                        await asyncio.sleep(0.001)

                async def receive_results():
                    nonlocal processed_frame, last_results, connection_status
                    while True:
                        try:
                            # 타임아웃 설정하여 무한 대기 방지 (서버 처리 시간 고려하여 60초로 증가)
                            message = await asyncio.wait_for(websocket.recv(), timeout=60.0)

                            if isinstance(message, str):
                                # JSON 결과 메시지 처리
                                try:
                                    data = json.loads(message)
                                    if data.get("type") == "results":
                                        last_results = data.get("data", {})
                                        processing_time = data.get("processing_time", -1)
                                        logging.info(f"결과 수신: {len(last_results.get('violations', []))} violations, 처리 시간: {processing_time}s")
                                    elif data.get("type") == "error":
                                        logging.error(f"서버 오류 수신: {data.get('message')}")
                                    # elif msg.data == 'pong': # 서버로부터 pong 확인
                                    #    logging.debug("Pong 수신")
                                    else:
                                         logging.debug(f"기타 텍스트 메시지 수신: {data.get('type')}")
                                except json.JSONDecodeError:
                                     logging.warning(f"수신된 JSON 파싱 불가: {message[:100]}")
                                except Exception as e:
                                    logging.error(f"결과 메시지 처리 중 오류: {e}")


                            elif isinstance(message, bytes):
                                # 처리된 이미지 프레임(바이너리) 처리
                                try:
                                    nparr = np.frombuffer(message, np.uint8)
                                    img_decode = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                    if img_decode is None:
                                         logging.warning("수신된 이미지 디코딩 실패")
                                         # 오류 프레임을 표시하도록 None 대신 빈 이미지 설정
                                         processed_frame = np.zeros((height, width, 3), dtype=np.uint8)
                                         cv2.putText(processed_frame, "Decode Error", (50, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                    else:
                                        processed_frame = img_decode # 성공 시 업데이트
                                except Exception as e:
                                    logging.error(f"수신된 이미지 처리 중 오류: {e}")
                                    processed_frame = np.zeros((height, width, 3), dtype=np.uint8)
                                    cv2.putText(processed_frame, "Img Proc Error", (50, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                        except asyncio.TimeoutError:
                            # 타임아웃 발생 시 서버가 살아있는지 확인
                            logging.debug("결과 수신 타임아웃. 서버 상태 확인 중...")
                            try:
                                # ping 메시지 전송하여 연결 확인
                                await websocket.ping()
                            except (websockets.ConnectionClosed, OSError):
                                logging.warning("서버 연결이 끊어진 것으로 확인됨.")
                                connection_status = "연결 끊김"
                                processed_frame = None
                                break
                            continue # 연결이 살아있으면 계속 수신 시도
                        except websockets.ConnectionClosed:
                            logging.warning("결과 수신 중 연결이 끊겼습니다.")
                            connection_status = "연결 끊김"
                            processed_frame = None # 연결 끊김 표시
                            break
                        except Exception as e:
                            logging.error(f"결과 수신 중 오류: {e}")
                            connection_status = f"수신 오류: {e}"
                            await asyncio.sleep(1)

                # 송신 및 수신 작업을 동시에 실행
                send_task = asyncio.create_task(send_frames())
                receive_task = asyncio.create_task(receive_results())

                # --- 결과 표시 루프 (옵션) ---
                import os
                show_window = os.getenv('SHOW_WINDOW', '0') == '1'
                window_name = "Processed Stream from Server"
                user_requested_exit = False  # 사용자가 종료 요청했는지 확인
                
                if show_window:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # 창 크기 조절 가능하도록

                # 작업들을 감시하면서 창 표시도 함께 처리
                while not send_task.done() and not receive_task.done():
                    if show_window:
                        display_frame = None
                        if processed_frame is not None:
                             # 수신된 프레임 크기에 맞춰 표시
                             display_frame = processed_frame.copy()
                        else:
                            # 연결 끊김 또는 오류 시 표시 (카메라 해상도 기준)
                            display_frame = np.zeros((height, width, 3), dtype=np.uint8)
                            cv2.putText(display_frame, connection_status, (50, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                        # 화면에 표시
                        cv2.imshow(window_name, display_frame)

                        # 'q' 키를 누르면 종료
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27:  # 'q' 또는 ESC
                            user_requested_exit = True
                            send_task.cancel()
                            receive_task.cancel()
                            break

                    # 짧은 대기로 CPU 사용량 줄이기
                    await asyncio.sleep(0.01)

                # 작업 취소 기다리기
                logging.info("작업 취소 중...")
                await asyncio.gather(send_task, receive_task, return_exceptions=True)
                
                # 사용자가 종료 요청했으면 외부 루프도 종료
                if user_requested_exit:
                    break
                # 그 외의 경우(연결 끊김 등)는 재연결을 위해 continue
                continue



        except websockets.exceptions.ConnectionClosedError as e:
            logging.warning(f"서버 연결 실패 ({e}). 5초 후 재시도합니다.")
            connection_status = "연결 실패. 재시도 중..."
            processed_frame = None # 연결 끊김 표시
            # 창이 열려있는지 확인하고 상태 표시 (표시 모드에서만)
            import os
            if os.getenv('SHOW_WINDOW', '0') == '1':
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
                     error_display = np.zeros((height, width, 3), dtype=np.uint8)
                     cv2.putText(error_display, connection_status, (50, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                     cv2.imshow(window_name, error_display)
                     key = cv2.waitKey(1) & 0xFF
                     if key == ord('q') or key == 27: break
                else: # 창이 닫혔으면 종료
                     logging.info("표시 창이 닫혔습니다. 종료합니다.")
                     break
            await asyncio.sleep(5) # 재시도 대기 시간
        except asyncio.CancelledError:
            logging.info("클라이언트 작업 취소됨.")
            break
        except Exception as e:
            logging.error(f"클라이언트 실행 중 예상치 못한 오류: {e}", exc_info=True)
            break

    cap.release()
    cv2.destroyAllWindows()
    logging.info("클라이언트 종료.")

if __name__ == "__main__":
    try:
        asyncio.run(run_client())
    except KeyboardInterrupt:
        logging.info("사용자에 의해 클라이언트 종료.")
