import cv2
import numpy as np
import os
from pathlib import Path
import yaml

# --- 配置参数 ---
BASE_DIR = Path('sz_data332')          # 原始数据集根目录
OUTPUT_DIR = Path('sz_data332_3ch_sym_reg')  # 输出目录
TARGET_DIRS = ['images/train', 'images/val']  # 要处理的子目录


# ======================
#   图像配准函数（ECC）
# ======================
def register_images(fixed, moving, warp_mode=cv2.MOTION_AFFINE,
                    number_of_iterations=500, termination_eps=1e-6):
    """
    使用 OpenCV ECC 算法进行图像配准。
    fixed: 原图（基准）
    moving: 镜像图（待对齐）
    返回: 对齐后的 moving 图像
    """
    fixed_f = fixed.astype(np.float32)
    moving_f = moving.astype(np.float32)

    # 初始化变换矩阵
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # 终止条件
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        number_of_iterations,
        termination_eps
    )

    try:
        cc, warp_matrix = cv2.findTransformECC(
            fixed_f, moving_f, warp_matrix, warp_mode, criteria
        )
        sz = fixed.shape
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            aligned = cv2.warpPerspective(
                moving, warp_matrix, (sz[1], sz[0]),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            )
        else:
            aligned = cv2.warpAffine(
                moving, warp_matrix, (sz[1], sz[0]),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            )
        return aligned

    except cv2.error as e:
        print(f"[ECC Warning] Registration failed: {e}")
        return moving  # 返回未配准图像


# ===========================
#  创建三通道对称输入函数
# ===========================
def create_3ch_symmetry_input(image_path):
    """生成 [原图, 配准镜像图, 绝对差值图] 的 3 通道输入"""
    img_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f"Warning: Could not read image {image_path}")
        return None

    # Step 1: 镜像
    img_flipped = cv2.flip(img_gray, 1)

    # Step 2: ECC 配准
    img_aligned = register_images(img_gray, img_flipped)

    # Step 3: 差值图
    img_diff = cv2.absdiff(img_gray, img_aligned)

    # Step 4: 合成为3通道
    input_3ch = np.stack([img_gray, img_aligned, img_diff], axis=-1)

    return input_3ch


# ===========================
#   批量处理整个数据集
# ===========================
def batch_process():
    print(f"开始生成带配准的 3 通道对称特征数据集到 {OUTPUT_DIR}")

    # 创建输出目录结构
    for target_dir in TARGET_DIRS:
        source_path = BASE_DIR / target_dir
        output_path = OUTPUT_DIR / target_dir
        output_path.mkdir(parents=True, exist_ok=True)

        # 标签路径
        labels_path = BASE_DIR / 'labels' / target_dir.split('/')[-1]
        output_labels_path = OUTPUT_DIR / 'labels' / target_dir.split('/')[-1]
        output_labels_path.mkdir(parents=True, exist_ok=True)

        # 批量处理图像
        for image_path in source_path.iterdir():
            if image_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff']:
                img_3ch = create_3ch_symmetry_input(image_path)

                if img_3ch is not None:
                    output_file = output_path / image_path.with_suffix('.png').name
                    cv2.imwrite(str(output_file), img_3ch)

                    # 复制标签
                    label_filename = image_path.with_suffix('.txt').name
                    try:
                        os.link(labels_path / label_filename,
                                output_labels_path / label_filename)
                    except FileExistsError:
                        pass
                    except FileNotFoundError:
                        print(f"Warning: Label file {label_filename} not found for {image_path.name}")

    # 修改 data.yaml
    data_yaml_path = BASE_DIR / 'data.yaml'
    output_data_yaml_path = OUTPUT_DIR / 'data.yaml'

    if data_yaml_path.exists():
        with open(data_yaml_path, 'r') as f:
            data_cfg = yaml.safe_load(f)

        # 转换为绝对路径
        data_cfg['train'] = str((OUTPUT_DIR / 'images/train').resolve())
        data_cfg['val']   = str((OUTPUT_DIR / 'images/val').resolve())

        with open(output_data_yaml_path, 'w') as f:
            yaml.safe_dump(data_cfg, f)


    print("✅ 3 通道 + 配准 数据集生成完毕！")


# ===========================
#          主入口
# ===========================
if __name__ == '__main__':
    batch_process()
