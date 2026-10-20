from pathlib import Path
import os
import argparse
import sys
import contextlib

# 确保工作目录是脚本所在目录
HERE = Path(__file__).resolve().parent
os.chdir(HERE)

def add_torch_safe_globals():
    """解决某些情况下加载模型时的pickle安全报错"""
    try:
        import torch
        # 尝试导入 DetectionModel，如果你的 tasks.py 修改正确，这里应该能正常工作
        from ultralytics.nn.tasks import DetectionModel
        with contextlib.suppress(Exception):
            torch.serialization.add_safe_globals([DetectionModel])
    except Exception:
        pass

def parse_args():
    p = argparse.ArgumentParser(description='Run Ultralytics Training')
    # 数据集配置文件
    p.add_argument('--data', default='my_data.yaml', help='path to dataset yaml')
    # 模型文件 (可以是 .pt 或 .yaml)
    p.add_argument('--model', default='yolov8n.yaml', help='model file (yaml or pt)')
    
    # 训练参数
    p.add_argument('--epochs', type=int, default=300, help='number of epochs')
    p.add_argument('--batch', type=int, default=16, help='batch size')
    p.add_argument('--imgsz', type=int, default=640, help='image size')
    p.add_argument('--workers', type=int, default=8, help='number of workers')
    p.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    p.add_argument('--project', default='runs/detect', help='save to project/name')
    p.add_argument('--name', default=None, help='save to project/name')
    p.add_argument('--resume', action='store_true', help='resume training from last checkpoint')
    
    return p.parse_args()

def main():
    args = parse_args()

    # 1. 尝试添加安全全局变量 (防止加载旧权重报错)
    add_torch_safe_globals()

    try:
        from ultralytics import YOLO
    except ImportError:
        print('❌ Error: ultralytics package not found. Please run "pip install -e ."')
        sys.exit(1)

    print(f"🚀 Running {args.model} on device='{args.device if args.device else 'auto'}'")

    try:
        # 2. 加载模型
        # 如果是 .yaml (如 yolov8-se.yaml)，这里会初始化新结构
        # 如果是 .pt，这里会加载预训练权重
        model = YOLO(args.model)

        # 3. 确定运行名称 (方便消融实验对比)
        # 如果命令行没指定 name，就用模型文件名 (例如 yolov8-se)
        run_name = args.name if args.name else Path(args.model).stem

        # 4. 开始训练
        model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            workers=args.workers,
            project=args.project,
            name=run_name,      # 结果会保存在 runs/detect/yolov8-se/
            exist_ok=True,      # 允许覆盖同名文件夹
            resume=args.resume,
            patience=50,        # Early stopping
        )

    except Exception as e:
        print('\n❌ Training failed:')
        print(e)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()