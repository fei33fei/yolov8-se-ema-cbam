from pathlib import Path
import csv
import zipfile
import sys

ROOT = Path(__file__).resolve().parent.parent
csv_file = ROOT / 'scripts' / 'missing_empty_labels.csv'
out_dir = ROOT / 'scripts' / 'labeling_packages'
out_dir.mkdir(parents=True, exist_ok=True)

if not csv_file.exists():
    print('CSV not found:', csv_file)
    sys.exit(1)

rows = []
with open(csv_file, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

train_missing = [r for r in rows if r['split'] == 'train' and r['type'] == 'missing']
val_empty = [r for r in rows if r['split'] == 'val' and r['type'] == 'empty']

def make_zip(name, items):
    if not items:
        print(f'No items for {name}, skipping')
        return None
    zip_path = out_dir / f'{name}.zip'
    with zipfile.ZipFile(zip_path, 'w') as z:
        for r in items:
            img = Path(r['image_path'])
            if img.exists():
                z.write(img, arcname=img.name)
    print('Wrote', zip_path, 'with', len(items), 'images')
    return zip_path

zip1 = make_zip('cvat_train_missing', train_missing[:2000])  # limit to 2000 by default
zip2 = make_zip('cvat_val_empty', val_empty[:2000])

# Create a LabelImg list (plain text of image paths) to help open images in LabelImg
labelimg_list = out_dir / 'labelimg_images.txt'
with open(labelimg_list, 'w', encoding='utf-8') as f:
    for r in train_missing + val_empty:
        f.write(r['image_path'] + '\n')

print('\nLabelImg list saved to:', labelimg_list)
print('Labeling packages saved to:', out_dir)
