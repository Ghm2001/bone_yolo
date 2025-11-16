1. <!-- #huafen.py -->
[text](huafen.py)
划分数据集，train：val = 8：2
INPUT_IMAGES_DIR（所有病例图片.png格式）
INPUT_LABELS_DIR（所有病例标注.txt格式）

2. <!-- (generate_3ch_symmetry_ECC.py) -->
[text](generate_3ch_symmetry_ECC.py)
三通道构建+ecc配准

3. train_yolo.py
   训练启动文件

4. train_config_3ch_sym.yaml
配置文件（参数修改）

5. 训练运行命令：
CUDA_VISIBLE_DEVICES=4 python train_yolo.py --cfg train_config_3ch_sym.yaml

二. 原始训练不加三通道
train_config.yaml（配置文件）
运行命令：train_config.yaml

