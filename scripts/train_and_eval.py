import argparse
from pathlib import Path
import json
import time

ROOT = Path(__file__).resolve().parent.parent

def run(data='my_data.yaml', model='yolov8n.pt', epochs=50, batch=16, device='cpu', project=None, name=None):
    from ultralytics import YOLO
    ts = time.strftime('%Y%m%d-%H%M%S')
    project = project or ROOT / 'runs' / 'detect'
    name = name or f'run_{ts}'
    print('Training with', dict(data=data, model=model, epochs=epochs, batch=batch, device=device))
    y = YOLO(model)
    # train
    y.train(data=data, epochs=epochs, batch=batch, device=device, project=str(project), name=name, exist_ok=True)

    # locate weights
    weights_dir = Path(project) / name / 'weights'
    best = weights_dir / 'best.pt'
    last = weights_dir / 'last.pt'
    best_path = str(best) if best.exists() else None
    last_path = str(last) if last.exists() else None

    # run validation using best.pt if exists
    val_results = None
    if best_path:
        print('Running validation on', best_path)
        y_best = YOLO(best_path)
        try:
            val_results = y_best.val(data=data, device=device)
        except Exception as e:
            print('Validation failed:', e)

    # save a run summary
    summary = {
        'data': data,
        'model': model,
        'epochs': epochs,
        'batch': batch,
        'device': device,
        'project': str(project),
        'name': name,
        'best': best_path,
        'last': last_path,
        'val_results': str(val_results)[:100] if val_results is not None else None
    }
    out = Path(project) / name / 'run_summary.json'
    out.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('Wrote run summary to', out)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='my_data.yaml')
    p.add_argument('--model', default='yolov8n.pt')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--device', default='cpu')
    p.add_argument('--project', default=None)
    p.add_argument('--name', default=None)
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run(data=args.data, model=args.model, epochs=args.epochs, batch=args.batch, device=args.device, project=args.project, name=args.name)
