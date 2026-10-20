from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parent.parent
infile = ROOT / 'scripts' / 'missing_empty_labels.txt'
outfile = ROOT / 'scripts' / 'missing_empty_labels.csv'

if not infile.exists():
    print('Input file not found:', infile)
    raise SystemExit(1)

rows = []
cur_split = None
cur_kind = None
for line in infile.read_text(encoding='utf-8').splitlines():
    line = line.strip()
    if not line:
        continue
    if line.startswith('##'):
        parts = line.split()
        # expect: ## train missing (N)
        if len(parts) >= 3:
            cur_split = parts[1]
            cur_kind = parts[2]
        continue
    if cur_split and cur_kind:
        fname = line
        img_path = str((ROOT / 'datasets' / 'NEU-DET' / 'images' / cur_split / fname).as_posix())
        label_path = str((ROOT / 'datasets' / 'NEU-DET' / 'labels' / cur_split / (Path(fname).stem + '.txt')).as_posix())
        rows.append({'split': cur_split, 'type': cur_kind, 'filename': fname, 'image_path': img_path, 'label_path': label_path})

with open(outfile, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['split', 'type', 'filename', 'image_path', 'label_path'])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print('Wrote CSV with', len(rows), 'rows to', outfile)
