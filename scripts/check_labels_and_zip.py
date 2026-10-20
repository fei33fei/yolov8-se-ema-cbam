from pathlib import Path
import zipfile
import sys

ROOT = Path(__file__).resolve().parent.parent
ds = ROOT / 'datasets' / 'NEU-DET'
if not ds.exists():
    print('Dataset path not found:', ds)
    sys.exit(1)

results = {}
for split in ('train', 'val'):
    img_dir = ds / 'images' / split
    lbl_dir = ds / 'labels' / split
    imgs = [p for p in img_dir.glob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    missing = []
    empty = []
    for p in imgs:
        txt = lbl_dir / (p.stem + '.txt')
        if not txt.exists():
            missing.append(p.name)
        else:
            try:
                if txt.stat().st_size == 0:
                    empty.append(p.name)
            except Exception:
                empty.append(p.name)
    results[split] = {'images': len(imgs), 'missing': missing, 'empty': empty}

print('Label summary for datasets/NEU-DET:')
for s, v in results.items():
    print(f"- {s}: images={v['images']}, missing_labels={len(v['missing'])}, empty_labels={len(v['empty'])}")
    if v['missing']:
        print('  sample missing:', v['missing'][:20])
    if v['empty']:
        print('  sample empty:', v['empty'][:20])

# Zip visualization images
viz_dir = ROOT / 'runs' / 'detect' / 'train'
if not viz_dir.exists():
    print('\nNo visualization directory found at', viz_dir)
    sys.exit(0)
zip_path = viz_dir / 'train_images_for_review.zip'
with zipfile.ZipFile(zip_path, 'w') as z:
    for img in viz_dir.glob('*.jpg'):
        z.write(img, arcname=img.name)
print('\nSaved visualization images zip to:', zip_path)

print('\nYou can open the zip file or `runs/detect/train` to view generated images (train_batch*.jpg).')

# Also write full missing/empty lists to a file for manual review
out = ROOT / 'scripts' / 'missing_empty_labels.txt'
with open(out, 'w', encoding='utf-8') as f:
    for split in ('train', 'val'):
        f.write(f'## {split} missing ({len(results[split]["missing"])})\n')
        for name in results[split]['missing']:
            f.write(name + '\n')
        f.write(f'## {split} empty ({len(results[split]["empty"])})\n')
        for name in results[split]['empty']:
            f.write(name + '\n')

print('\nWrote missing/empty label lists to:', out)
