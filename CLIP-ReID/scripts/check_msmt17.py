from pathlib import Path
import re
from collections import Counter

# 실제 경로에 맞춰 사용
DATA_ROOT = Path("/home/dicia/JWChu/datasets/msmt17")

folders = ["bounding_box_train", "bounding_box_test", "query"]

# MSMT17 파일명은 버전에 따라 형식이 다를 수 있어서 넓게 잡음
cam_patterns = [
    re.compile(r"_c(\d+)"),
    re.compile(r"_(\d{2})_"),   # 예: 0001_0001_01_0001.jpg 같은 형식 대응
]

print("DATA_ROOT:", DATA_ROOT)
print("exists:", DATA_ROOT.exists())

for folder in folders:
    path = DATA_ROOT / folder
    files = sorted(list(path.rglob("*.jpg")) + list(path.rglob("*.png")))

    print(f"\n[{folder}]")
    print("path:", path)
    print("exists:", path.exists())
    print("num_images:", len(files))

    pids = []
    cams = []

    for f in files:
        # pid는 상위 폴더명이 숫자면 그걸 우선 사용
        if f.parent.name.isdigit():
            pids.append(int(f.parent.name))
        else:
            m_pid = re.match(r"([-\d]+)", f.name)
            if m_pid:
                pids.append(int(m_pid.group(1)))

        cam = None
        for pat in cam_patterns:
            m_cam = pat.search(f.name)
            if m_cam:
                cam = int(m_cam.group(1))
                break
        if cam is not None:
            cams.append(cam)

    print("num_ids:", len(set(pids)))
    print("num_cameras:", len(set(cams)))
    print("camera_count:", dict(sorted(Counter(cams).items())))

    print("sample files:")
    for s in files[:10]:
        print(" ", s.relative_to(DATA_ROOT))