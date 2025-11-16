# 随机划分训练集和测试集
import os
import shutil
from sklearn.model_selection import train_test_split

# --- 路径配置 ---
INPUT_IMAGES_DIR = '/home/gonghanmei/datasets/shen_data/png332'
INPUT_LABELS_DIR = '/home/gonghanmei/datasets/shen_data/processed_labels/processed_labels'
OUTPUT_DATA_DIR = '/home/gonghanmei/project/yolo/sz_data332'

# --- 分割比例配置 ---
# 仅划分训练集和测试集：80% 训练集, 20% 测试集
TEST_SIZE = 0.20  

DATA_YAML_TEMPLATE = """
# data.yaml for YOLO training
# 注意：路径是相对于训练脚本运行位置的相对路径
path: {final_data_folder}
train: images/train
val: images/val

# 类别信息
names:
  0: bone_metastasis # 请确保类别名称正确
"""

def split_data_by_case(input_img_dir, input_lbl_dir, output_dir):
    print("开始数据集分割和结构化 (以病例为单位)...")
    
    # 查找所有图像文件，并提取唯一的病例 ID
    all_img_files = [f for f in os.listdir(input_img_dir) if f.endswith('.png')]
    
    if not all_img_files:
        print(f"错误：在 {input_img_dir} 文件夹中找不到任何图像文件 (.png)。请检查路径。")
        return
    
    # 获取唯一的病例 ID (去除 _front.png 或 _back.png 等后缀)
    unique_case_ids = set()
    for filename in all_img_files:
        # 假设文件名格式是 "CASE_ID_SUFFIX.png"
        # 尝试匹配常见的后缀并去除，以获得病例 ID
        case_id = filename.replace('.png', '')
        if '_front' in case_id:
            case_id = case_id.replace('_front', '')
        elif '_back' in case_id:
            case_id = case_id.replace('_back', '')
        # 如果是之前合并后的文件，可能命名为 "CASE_ID_combined.png"
        elif '_combined' in case_id:
            case_id = case_id.replace('_combined', '')
        else:
            # 如果文件名没有后缀（例如之前手动合并后的文件），则直接使用
            pass
            
        # 最终使用原始文件名作为 ID（因为它包含了正反面的信息）
        # 但是为了确保 front 和 back 被一起处理，我们需要一个通用的 ID
        # 让我们使用一种更健壮的方式：查找所有文件名中没有 _front/_back 的部分
        
        # 重新定义：我们将病例 ID 定义为文件名中除了 "_front" 或 "_back" 的部分
        base_id = filename.replace('_front.png', '').replace('_back.png', '').replace('.png', '')
        if not base_id.endswith('_'): # 避免错误删除
            unique_case_ids.add(base_id)


    all_unique_ids = sorted(list(unique_case_ids))
    
    if not all_unique_ids:
        print("错误：无法从文件名中解析出唯一的病例 ID。请检查您的文件名格式。")
        return

    print(f"找到 {len(all_unique_ids)} 个唯一的病例 ID。")
    
    # --- 1. 分割: 训练集 (80%) 和 测试集 (20%) ---
    train_ids, test_ids = train_test_split(
        all_unique_ids, 
        test_size=TEST_SIZE, 
        random_state=42  # 确保每次运行结果一致
    )
    
    splits = {
        'train': train_ids,
        'val': test_ids
    }
    
    # 清理旧的输出目录并创建新结构
    if os.path.exists(output_dir):
        print(f"警告：正在清空输出目录 {output_dir}...")
        for sub_dir in ['images', 'labels']:
            full_path = os.path.join(output_dir, sub_dir)
            if os.path.exists(full_path):
                shutil.rmtree(full_path)

    for split_name in splits.keys():
        os.makedirs(os.path.join(output_dir, 'images', split_name), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split_name), exist_ok=True)
        
    # --- 2. 移动文件到新目录 ---
    for split_name, ids in splits.items():
        for case_id in ids:
            
            # 找到当前 case_id 对应的所有图像和标签文件
            # 考虑到可能存在 _front 和 _back
            file_suffixes = ['_front', '_back', ''] # 包含空后缀以处理手动合并后的文件
            moved_count = 0
            
            for suffix in file_suffixes:
                img_filename = f"{case_id}{suffix}.png"
                label_filename = f"{case_id}{suffix}.txt"
                
                src_img_path = os.path.join(input_img_dir, img_filename)
                src_lbl_path = os.path.join(input_lbl_dir, label_filename)
                
                # 检查图像文件是否存在
                if os.path.exists(src_img_path):
                    
                    # 目标路径
                    dst_img_path = os.path.join(output_dir, 'images', split_name, img_filename)
                    dst_lbl_path = os.path.join(output_dir, 'labels', split_name, label_filename)
                    
                    # 复制图像文件
                    shutil.copy(src_img_path, dst_img_path)
                    
                    # 复制标签文件（处理空标签/文件不存在的情况）
                    if os.path.exists(src_lbl_path):
                        shutil.copy(src_lbl_path, dst_lbl_path)
                    else:
                         # 创建一个空标签文件
                        with open(dst_lbl_path, 'w') as f:
                            pass
                    
                    moved_count += 1
            
            if moved_count == 0:
                 print(f"警告: 未找到 {case_id} 对应的任何文件，可能ID提取有误。")
                
        print(f"已为 {split_name} 移动 {len(ids)} 个病例ID，总计 {moved_count} 个文件。")

def create_yaml_file(output_path):
    """
    创建并写入 data.yaml 文件。
    """
    # 这里的 final_data_folder 是相对于训练脚本运行位置的相对路径
    yaml_content = DATA_YAML_TEMPLATE.format(final_data_folder=os.path.basename(output_path))
    yaml_path = os.path.join(output_path, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"\n已在 {yaml_path} 生成 data.yaml 文件。")

# --- 主程序入口 ---
if __name__ == '__main__':
    try:
        # 使用你新的数据，这些数据是手动对齐后的（可能是 _combined.png 或原始的 _front/_back.png）
        # 如果你手动合并为一张图片，请确保你的文件命名不再包含 _front/_back，而是直接是 CASE_ID.png
        # 脚本将尝试匹配所有可能性
        split_data_by_case(INPUT_IMAGES_DIR, INPUT_LABELS_DIR, OUTPUT_DATA_DIR)
        create_yaml_file(OUTPUT_DATA_DIR)
        print("\n数据集分割和结构化完成！")
        print(f"新的数据集已在 '{OUTPUT_DATA_DIR}' 目录中准备就绪。")
    except Exception as e:
        print(f"脚本执行失败: {e}")