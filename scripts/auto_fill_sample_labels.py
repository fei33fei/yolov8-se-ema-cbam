from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
ds = ROOT / 'datasets' / 'NEU-DET'
lists_file = ROOT / 'scripts' / 'missing_empty_labels.txt'

def read_lists(path):
    data = {'train': {'missing': [], 'empty': []}, 'val': {'missing': [], 'empty': []}}
    if not path.exists():
        print('missing_empty_labels.txt not found, run check_labels_and_zip.py first')
        sys.exit(1)
    cur = None
    kind = None
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.startswith('##'):
            # format: ## train missing (N)
            parts = line.split()
            cur = parts[1]
            kind = parts[2]
            continue
        if not line.strip():
            continue
        if cur and kind:
            data[cur][kind].append(line.strip())
    return data

def ensure_dir(p):
    p.mkdir(parents=True, exist_ok=True)

def fill_labels(n_per_split=20):
    data = read_lists(lists_file)
    created = []
    for split in ('train', 'val'):
        img_dir = ds / 'images' / split
        lbl_dir = ds / 'labels' / split
        ensure_dir(lbl_dir)
        # fill from missing first
        miss = data[split]['missing'][:n_per_split]
        empty = data[split]['empty'][:n_per_split]
        for name in miss + empty:
            # create txt with full-image bbox for class 0
            txt = lbl_dir / (Path(name).stem + '.txt')
            # backup existing empty file
            if txt.exists() and txt.stat().st_size == 0:
                bak = txt.with_suffix(txt.suffix + '.bak')
                try:
                    txt.rename(bak)
                except Exception:
                    pass
            content = '0 0.5 0.5 1.0 1.0\n'
            txt.write_text(content, encoding='utf-8')
            created.append(str(txt))
    print(f'Wrote {len(created)} example label files (class 0 full-image bbox).')
    print('Sample created files:', created[:20])

if __name__ == '__main__':
    n = 20
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except Exception:
            pass
    fill_labels(n_per_split=n)
