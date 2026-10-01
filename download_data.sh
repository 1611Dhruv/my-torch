#!/usr/bin/env bash
# Download TinyStories V2, the CS336 OpenWebText sample, and the Smithsonian
# butterfly images.
#
# Usage:
#   ./download_data.sh [tinystories|owt|butterflies|all] [DATA_DIR]
#   DATA_DIR defaults to $DATA_DIR, else ./data
#
# Butterflies (Kaggle, 1000 images, all PNG ~512px wide) end up as:
#   butterflies/images/*.png              originals
#   butterflies/metadata.csv              species, taxonomy, etc. + file column
#   butterflies/images_<SIZE>/*_<SIZE>x<SIZE>.<FMT>   resized, square
# Knobs:
#   BUTTERFLY_SIZE=128 ./download_data.sh butterflies   # default 64
#   BUTTERFLY_MODE=crop ./download_data.sh butterflies  # pad (default) | crop
#   BUTTERFLY_FORMAT=jpg ./download_data.sh butterflies # png (default) | jpg
# JPEGs are baseline (not progressive), 4:2:0 subsampling, quality 95 -- the
# simplest flavour for a hand-written decoder.
# Resizing needs Pillow (use PYTHON=~/venv/bin/python on Ubuntu); without it
# you still get the original PNGs.
#
set -euo pipefail

WHICH="${1:-all}"
DATA_DIR="${2:-${DATA_DIR:-./data}}"
PYTHON="${PYTHON:-python3}"
BUTTERFLY_SIZE="${BUTTERFLY_SIZE:-64}"
BUTTERFLY_MODE="${BUTTERFLY_MODE:-pad}"
BUTTERFLY_FORMAT="${BUTTERFLY_FORMAT:-png}"
case "$BUTTERFLY_FORMAT" in
  png|jpg) ;;
  jpeg) BUTTERFLY_FORMAT=jpg ;;
  *) echo "BUTTERFLY_FORMAT must be png or jpg" >&2; exit 2 ;;
esac

HF="https://huggingface.co/datasets"
TINY_FILES=(
  "$HF/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
  "$HF/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
)
OWT_FILES=(
  "$HF/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz"
  "$HF/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz"
)
BUTTERFLY_URL="https://www.kaggle.com/api/v1/datasets/download/thedevastator/smithsonian-butterfly-dataset"

# Rough space needed in GB (OWT needs room for the .gz and the unzipped file at
# once; butterflies need the ~0.33 GB zip plus ~0.25 GB of PNGs).
URLS=()
BUTTERFLIES=0
case "$WHICH" in
  tinystories) URLS=("${TINY_FILES[@]}"); NEED_GB=3 ;;
  owt)         URLS=("${OWT_FILES[@]}");  NEED_GB=17 ;;
  butterflies) BUTTERFLIES=1;             NEED_GB=1 ;;
  all)         URLS=("${TINY_FILES[@]}" "${OWT_FILES[@]}"); BUTTERFLIES=1; NEED_GB=21 ;;
  *) echo "Usage: $0 [tinystories|owt|butterflies|all] [DATA_DIR]" >&2; exit 2 ;;
esac

command -v wget >/dev/null || { echo "wget not found: sudo apt install -y wget" >&2; exit 1; }

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

avail_gb=$(df -Pk . | awk 'NR==2 {print int($4 / 1024 / 1024)}')
echo "Target: $(pwd)  (${avail_gb} GB free, ~${NEED_GB} GB needed)"
if (( avail_gb < NEED_GB )); then
  echo "Not enough space. Grow the disk or pass a bigger DATA_DIR (e.g. /mnt/nvme/data)." >&2
  exit 1
fi

for url in "${URLS[@]+"${URLS[@]}"}"; do
  file="$(basename "$url")"
  final="${file%.gz}"

  if [[ -s "$final" ]]; then
    echo "[x] $final already present, skipping"
    continue
  fi

  echo "↓ $file"
  wget -c -q --show-progress "$url" -O "$file"

  if [[ "$file" == *.gz ]]; then
    echo "  unzipping $file"
    gunzip -f "$file"
  fi

  [[ -s "$final" ]] || { echo "[ ] $final is missing or empty after download" >&2; exit 1; }
done

if (( BUTTERFLIES )); then
  RESIZED="butterflies/images_${BUTTERFLY_SIZE}"
  ZIP="butterflies/smithsonian-butterfly-dataset.zip"
  mkdir -p butterflies/images

  if [[ -s butterflies/metadata.csv ]]; then
    echo "[x] butterflies/images already present, skipping download"
  else
    echo "↓ smithsonian-butterfly-dataset.zip"
    # The zip holds one train.csv with each PNG embedded as {'bytes': b'...'}.
    wget -q --show-progress "$BUTTERFLY_URL" -O "$ZIP"
    echo "  extracting images"
    "$PYTHON" - "$ZIP" butterflies <<'EOF'
import ast, csv, io, re, sys, zipfile

zip_path, out_dir = sys.argv[1], sys.argv[2]
csv.field_size_limit(sys.maxsize)
MAGIC = [(b"\x89PNG\r\n\x1a\n", "png"), (b"\xff\xd8\xff", "jpg"), (b"GIF8", "gif"),
         (b"RIFF", "webp"), (b"II*\x00", "tif"), (b"MM\x00*", "tif")]

def ext_for(data):
    return next((ext for magic, ext in MAGIC if data.startswith(magic)), "bin")

def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")[:60] or "unknown"

with zipfile.ZipFile(zip_path) as z:
    reader = csv.DictReader(io.TextIOWrapper(z.open("train.csv"), encoding="utf-8"))
    cols = [c for c in reader.fieldnames if c != "image"] + ["file"]
    with open(f"{out_dir}/metadata.csv.part", "w", newline="") as mf:
        writer = csv.DictWriter(mf, fieldnames=cols)
        writer.writeheader()
        counts, skipped = {}, 0
        for i, row in enumerate(reader):
            try:
                data = ast.literal_eval(row.pop("image"))["bytes"]
            except Exception:
                skipped += 1
                continue
            ext = ext_for(data)
            name = f"{i:04d}_{slug(row['scientific_name'] or row['name'])}.{ext}"
            with open(f"{out_dir}/images/{name}", "wb") as f:
                f.write(data)
            row["file"] = f"images/{name}"
            writer.writerow(row)
            counts[ext] = counts.get(ext, 0) + 1
print(f"  wrote {sum(counts.values())} images {counts}, skipped {skipped}")
EOF
    # Rename only on success, so a failed run is retried instead of skipped.
    mv butterflies/metadata.csv.part butterflies/metadata.csv
    rm -f "$ZIP"
  fi

  if compgen -G "$RESIZED/*_${BUTTERFLY_SIZE}x${BUTTERFLY_SIZE}.${BUTTERFLY_FORMAT}" >/dev/null; then
    echo "[x] $RESIZED/*.${BUTTERFLY_FORMAT} already present, skipping"
  elif ! "$PYTHON" -c "import PIL" 2>/dev/null; then
    echo "  skipping resize: $PYTHON lacks Pillow (pip install pillow, or set PYTHON=)" >&2
  else
    echo "  resizing to ${BUTTERFLY_SIZE}x${BUTTERFLY_SIZE} ($BUTTERFLY_MODE, $BUTTERFLY_FORMAT)"
    mkdir -p "$RESIZED"
    "$PYTHON" - butterflies "$BUTTERFLY_SIZE" "$BUTTERFLY_MODE" "$BUTTERFLY_FORMAT" "$RESIZED" <<'EOF'
import csv, os, sys
from PIL import Image, ImageOps

root, size, mode, fmt, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
with open(f"{root}/metadata.csv", newline="") as f:
    rows = list(csv.DictReader(f))

for r in rows:
    im = Image.open(f"{root}/{r['file']}").convert("RGB")
    if mode == "crop":
        im = ImageOps.fit(im, (size, size), Image.LANCZOS)
    else:  # pad with the image's own corner color so the bars blend in
        im = ImageOps.pad(im, (size, size), Image.LANCZOS, color=im.getpixel((0, 0)))
    stem = os.path.splitext(os.path.basename(r["file"]))[0]
    path = f"{out}/{stem}_{size}x{size}.{fmt}"
    if fmt == "jpg":
        im.save(path, "JPEG", quality=95, progressive=False, optimize=False, subsampling="4:2:0")
    else:
        im.save(path, "PNG")
print(f"  {len(rows)} images -> {out}/")
EOF
  fi
fi

echo
echo "Done:"
ls -lh -- *.txt 2>/dev/null || true
if [[ -d butterflies ]]; then
  for d in butterflies/images*; do
    echo "$d: $(ls "$d" | wc -l | tr -d ' ') files, $(du -sh "$d" | cut -f1)"
  done
fi
