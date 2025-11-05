import logging
from core import SafetySystem
from utils import setup_logging

def main():
    """애플리케이션의 메인 실행 함수."""
    setup_logging()
    logging.info("--- 지능형 통합 안전 관리 시스템 초기화 ---")

    try:
        safety_system = SafetySystem()
        safety_system.run()
    except SystemExit as e:
        logging.error(f"시스템 초기화 실패. 프로그램을 종료합니다. 메시지: {e}")
    except Exception as e:
        logging.critical(f"예상치 못한 오류가 발생했습니다: {e}", exc_info=True)
    finally:
        logging.info("시스템을 종료합니다.")

if __name__ == "__main__":
    main()