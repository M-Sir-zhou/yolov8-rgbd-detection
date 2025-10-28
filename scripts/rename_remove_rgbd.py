# 将文件名中的 "_rgbd" 移除，保留其余部分（包括数字）。默认 dry-run，使用 --apply 才会真正重命名。
import argparse
from pathlib import Path
import shutil
import sys

def safe_new_name(p: Path, new_stem: str):
    new_name = new_stem + p.suffix
    candidate = p.with_name(new_name)
    i = 1
    while candidate.exists():
        candidate = p.with_name(f"{new_stem}_{i}{p.suffix}")
        i += 1
    return candidate

def rename_remove_rgbd(root: Path, apply: bool = False, recursive: bool = True, exts=None):
    if exts is None:
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    files = list(root.rglob("*") if recursive else root.glob("*"))
    files = [p for p in files if p.is_file() and p.suffix.lower() in exts]
    changed = []
    for p in files:
        stem = p.stem
        if "_rgbd" not in stem:
            continue
        new_stem = stem.replace("_rgbd", "")
        # 清理可能产生的双下划线或首尾空格，但保留其他字符（包括数字、括号等）
        new_stem = new_stem.replace("__", "_").strip()
        if new_stem == "":
            # 不安全，跳过
            print("Skip unsafe rename (empty stem) for:", p.name)
            continue
        target = safe_new_name(p, new_stem)
        changed.append((p, target))
    if not changed:
        print("No files to rename under", root)
        return 0
    print(f"Found {len(changed)} files to rename. Sample:")
    for src, dst in changed[:20]:
        print(f"  {src.name} -> {dst.name}")
    if not apply:
        print("\nDry run (no files renamed). Re-run with --apply to perform changes.")
        return len(changed)
    # 执行重命名
    renamed = 0
    for src, dst in changed:
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            src.rename(dst)
            renamed += 1
            print("Renamed:", src.name, "->", dst.name)
        except Exception as e:
            # 尝试 copy + unlink 以处理跨驱动器等问题
            print("Rename failed for", src.name, "err:", e, "trying copy+unlink")
            try:
                shutil.copy2(src, dst)
                src.unlink()
                renamed += 1
                print("Copied+removed:", src.name, "->", dst.name)
            except Exception as e2:
                print("Failed to move file:", src, "error:", e2)
    print("Done. Total renamed:", renamed)
    return renamed

def main():
    parser = argparse.ArgumentParser(description="Remove '_rgbd' from image filenames under tennis-yolo/images")
    parser.add_argument('--root', '-r', default=r"d:\ProjectCode\PyCharm\ultralytics-main\datasets\tennis-yolo\images",
                        help="images root directory")
    parser.add_argument('--apply', action='store_true', help="actually perform renames (default: dry-run)")
    parser.add_argument('--no-recursive', dest='recursive', action='store_false', help="do not recurse into subfolders")
    parser.add_argument('--exts', nargs='*', help="file extensions to consider, e.g. .jpg .png")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print("Root not found:", root)
        sys.exit(1)
    exts = None
    if args.exts:
        exts = set(e.lower() if e.startswith('.') else f".{e.lower()}" for e in args.exts)
    rename_remove_rgbd(root, apply=args.apply, recursive=args.recursive, exts=exts)

if __name__ == "__main__":
    main()