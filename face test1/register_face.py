# register_face.py
import os
import shutil

# --- 설정 ---
# 새로 등록할 얼굴 이미지들이 들어있는 폴더 (예: ./new_faces/홍길동/사진1.jpg)
INPUT_DIR = "./new_faces"
# 얼굴 DB가 저장될 최종 경로
DB_PATH = "../image"
# 처리가 완료된 폴더가 이동될 경로
PROCESSED_DIR = "./processed_faces"


def main():
    """ new_faces 폴더의 하위 폴더(사람 이름)를 image DB 폴더로 이동시키는 역할만 수행합니다. """
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(DB_PATH, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    person_folders = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]
    if not person_folders:
        print(f"\n'{INPUT_DIR}' 폴더에 처리할 이름 폴더가 없습니다.")
        print(f"예: '{os.path.join(INPUT_DIR, '홍길동')}' 폴더를 만들고 그 안에 사진을 넣어주세요.")
        return

    print(f"\n총 {len(person_folders)}명의 인물 폴더를 DB로 이동합니다.")
    for person_name in person_folders:
        source_dir = os.path.join(INPUT_DIR, person_name)
        destination_dir = os.path.join(DB_PATH, person_name)

        print("-" * 30)
        print(f"▶ '{person_name}' 폴더를 처리 중...")

        if os.path.exists(destination_dir):
            print(f"  '{person_name}' DB 폴더가 이미 존재합니다. 파일들을 통합합니다.")
            for filename in os.listdir(source_dir):
                shutil.move(os.path.join(source_dir, filename), os.path.join(destination_dir, filename))
            os.rmdir(source_dir)
        else:
            shutil.move(source_dir, destination_dir)

        print(f"  ✅ '{person_name}' 폴더를 '{DB_PATH}'(으)로 성공적으로 이동/통합했습니다.")

    print("-" * 30)
    print("🎉 모든 폴더 이동 처리가 완료되었습니다!")
    print("이제 'build_database.py'를 실행하여 얼굴 인식 DB를 업데이트하세요.")


if __name__ == "__main__":
    main()