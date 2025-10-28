import os, sys
from pathlib import Path

dataset = Path(r"d:\\ProjectCode\\PyCharm\\ultralytics-main\\datasets\\tennis-yolo")
labels_dirs = [dataset / "labels" / "train", dataset / "labels" / "val", dataset / "labels" / "test"]
img_dirs = [dataset / "images" / "train", dataset / "images" / "val", dataset / "images" / "test"]

def check_one(label_path, img_path, nc):
    problems = []
    try:
        txt = label_path.read_text().strip()
    except Exception:
        return ["label_file_missing"], None
    if txt == "":
        return ["no_labels"], None
    lines = txt.splitlines()
    for i, l in enumerate(lines):
        parts = l.split()
        if len(parts) != 5:
            problems.append(f"line{i+1}_bad_format:{l}")
            continue
        cls, x, y, w, h = parts
        try:
            ci = int(float(cls))
            xf = float(x); yf = float(y); wf = float(w); hf = float(h)
        except Exception:
            problems.append(f"line{i+1}_parse_error:{l}")
            continue
        if ci < 0 or ci >= nc:
            problems.append(f"line{i+1}_class_out_of_range:{ci}")
        if not (0.0 <= xf <= 1.0 and 0.0 <= yf <= 1.0):
            problems.append(f"line{i+1}_center_out_of_range:{xf},{yf}")
        if not (0.0 < wf <= 1.0 and 0.0 < hf <= 1.0):
            problems.append(f"line{i+1}_wh_invalid:{wf},{hf}")
    # image size (optional)
    if img_path.exists():
        import cv2
        im = cv2.imread(str(img_path))
        if im is None:
            problems.append("image_cannot_read")
        else:
            h, w = im.shape[:2]
            # check whether any bbox normalized extents exceed image when denorm
            # skip heavy checks
    else:
        problems.append("image_missing")
    return problems, len(lines)

def main():
    # read nc from dataset yaml if exists
# ...existing code...
    # read nc from dataset yaml if exists
    nc = 1
    yamlf = dataset / "tennis-yolo.yaml"
    if yamlf.exists():
        import yaml
        text = None
        # 尝试多种常见编码读取
        for enc in ("utf-8", "utf-8-sig", "utf-16", "gbk", "latin-1"):
            try:
                with yamlf.open("r", encoding=enc) as f:
                    text = f.read()
                print(f"Loaded yaml with encoding: {enc}")
                break
            except Exception:
                continue
        if text is None:
            # 兜底：以二进制读取并用 replacement 解码，避免抛出 UnicodeDecodeError
            try:
                text = yamlf.read_bytes().decode("utf-8", errors="replace")
                print("Loaded yaml with binary fallback (utf-8 replace).")
            except Exception as e:
                print("读取 yaml 文件失败:", e)
                text = None

        d = {}
        if text:
            try:
                d = yaml.safe_load(text) or {}
            except Exception as e:
                print("yaml.safe_load 解析失败:", e)
                d = {}
        # 最终确定 nc，保证类型安全
        try:
            nc = int(d.get("nc", 1)) if isinstance(d, dict) else 1
        except Exception:
            nc = 1
        print("Detected nc from yaml:", nc)
# ...existing code...
    total_report = []
    for lab_dir, img_dir in zip(labels_dirs, img_dirs):
        if not lab_dir.exists():
            print("labels dir missing:", lab_dir)
            continue
        files = sorted(p for p in lab_dir.glob("*.txt"))
        print(f"Checking {lab_dir} ({len(files)} files)...")
        badcount = 0
        for f in files:
            img_name = f.stem
            # try common image extensions
            imgp = None
            for ext in (".jpg",".jpeg",".png",".bmp"):
                p = img_dir / (img_name + ext)
                if p.exists():
                    imgp = p; break
            problems, nlab = check_one(f, imgp or Path(""), nc)
            if problems and problems != ["no_labels"]:
                badcount += 1
                print(f"BAD {f.relative_to(dataset)}: {problems} labels={nlab} img={imgp}")
        print(f"{lab_dir.name}: {badcount} bad files")
    print("done")

if __name__ == '__main__':
    main()