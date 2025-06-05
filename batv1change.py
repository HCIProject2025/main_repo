import os

batv1_label_path = "data/Swing_Test.v1i.yolov5pytorch/valid/labels/"

for file in os.listdir(batv1_label_path):
    if not file.endswith(".txt"):
        continue

    fpath = os.path.join(batv1_label_path, file)
    with open(fpath) as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        cls = parts[0]
        # 클래스 번호 뒤집기: 0 → 1, 1 → 0
        if cls == "0":
            parts[0] = "1"
        elif cls == "1":
            parts[0] = "0"
        new_lines.append(" ".join(parts))

    with open(fpath, "w") as f:
        f.write("\n".join(new_lines) + "\n")

print("✅ bat.v1i 라벨 클래스 번호 뒤바꾸기 완료")
