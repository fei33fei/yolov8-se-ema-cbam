import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置字体和负号显示
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False


def create_figure(output_dir=r'D:\Fey\Desktop\ultralytics-main\ultralytics-main\runs'):
    # 数据 - EMA模型六类缺陷检测性能
    categories = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
    precision = [0.588, 0.725, 0.855, 0.713, 0.488, 0.630]
    recall = [0.397, 0.694, 0.797, 0.682, 0.587, 0.751]
    map50 = [0.483, 0.795, 0.896, 0.749, 0.481, 0.706]

    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 7))

    # 修改后的蓝紫色系配色（低饱和度）
    # Precision: 浅灰蓝色 (#8DA0CB) - 低饱和蓝
    # Recall: 淡紫色 (#B4A7D6) - 低饱和紫
    # mAP@50: 灰紫色 (#9370DB) - 稍深的低饱和紫
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#8DA0CB', edgecolor='black', linewidth=0.8, alpha=0.9)
    bars2 = ax.bar(x, recall, width, label='Recall', color='#B4A7D6', edgecolor='black', linewidth=0.8, alpha=0.9)
    bars3 = ax.bar(x + width, map50, width, label='mAP@50', color='#9370DB', edgecolor='black', linewidth=0.8, alpha=0.9)

    # 添加数值标签
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontname='Times New Roman')

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    ax.set_xlabel('Defect Category', fontsize=14, fontname='Times New Roman', fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontname='Times New Roman', fontweight='bold')
    ax.set_title('Six Types of Steel Defect Detection Performance (EMA Model)', fontsize=16, fontname='Times New Roman', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, fontname='Times New Roman')
    ax.legend(prop={'family': 'Times New Roman', 'size': 12}, loc='upper right')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    # 创建输出目录并保存图片
    try:
        os.makedirs(output_dir, exist_ok=True)
        # 写入测试文件以确认写权限
        test_path = os.path.join(output_dir, 'GEN_TEST.txt')
        with open(test_path, 'w', encoding='utf-8') as f:
            f.write('write_test')
        output_path = os.path.join(output_dir, '六类钢材缺陷检测性能对比图.png')
        print(f"保存目标路径: {output_path}")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存为: {output_path}")
    except Exception as e:
        print(f"保存图片时出错: {e}")
        raise
    return output_path


if __name__ == '__main__':
    create_figure()