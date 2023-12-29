
import argparse
from PIL import Image
from PIL.Image import Resampling

from pathlib import Path

from tqdm import tqdm


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Path to the root directory")
    parser.add_argument("-p", "--pattern", type=str, help="Pattern to match", default="*.png")
    parser.add_argument("-f", "--format", type=str, help="Target file format", default="webp")
    parser.add_argument("-r", "--recursive", action="store_true", help="Recursively search for files")
    parser.add_argument("--quality", type=int, help="Quality of the webp image", default=80)
    parser.add_argument("--lossy", action="store_true", help="Use lossy compression")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")

    parser.add_argument("--max-size", type=int, help="Clamp converted image width/height to this", default=None)

    parser.add_argument("--dry-run", action="store_true", help="Don't actually convert anything")

    args = parser.parse_args()

    if args.recursive:
        files = list(args.root.rglob(args.pattern))
    else:
        files = list(args.root.glob(args.pattern))

    old_size = 0
    new_size = 0

    for file in tqdm(files, desc="Converting", unit="files"):
        target = file.with_suffix(f".{args.format}")
        if not args.overwrite and target.is_file():
            raise FileExistsError(f"File {target} already exists. Use --overwrite to overwrite existing files.")

        if args.dry_run:
            print(f"Would convert {file} to {target}")
            continue

        with Image.open(file) as img:
            if args.max_size is not None:
                img.thumbnail((args.max_size, args.max_size), Resampling.BICUBIC)

            img.save(target, format=args.format, lossless=not args.lossy, quality=args.quality, method=6)

            old_size += file.stat().st_size
            new_size += target.stat().st_size

    size_diff = sizeof_fmt(old_size - new_size)
    old_size = sizeof_fmt(old_size)
    new_size = sizeof_fmt(new_size)

    print(f"Done! The converted images are {old_size} - {new_size} = {size_diff} smaller.")


if __name__ == "__main__":
    main()
