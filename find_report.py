find_report.py

# 完美的高亮版本
from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path
import re


# --- 路径配置 ---
vis_dir = Path("/home/gonghanmei/project/yolo/vis_1/cropped")
report_dir = Path("/home/wangnannan/data/spect/sz/reports/all")
save_dir = Path("/home/gonghanmei/project/yolo/vis_1/highlight")
save_dir.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------
# 🌟 字体修复
# ----------------------------------------------------
CHINESE_FONT_FILENAME = "simhei.ttf"
FONT_SIZE = 18

try:
    # 尝试加载中文字体，请确保 simhei.ttf 位于当前运行环境中或系统字体路径下
    font = ImageFont.truetype(CHINESE_FONT_FILENAME, FONT_SIZE)
    print(f"✅ 成功加载中文字体: {CHINESE_FONT_FILENAME}")
except Exception as e:
    font = ImageFont.load_default()
    print(f"❌ 无法加载 {CHINESE_FONT_FILENAME}，中文可能乱码：{e}")


# ----------------------------------------------------
# 🌟 寻找需要高亮的段落位置
# ----------------------------------------------------
def find_highlight_range(text):
    """
    返回需要高亮的文本起止位置 (start, end)
    """
    pattern = re.compile(
        r"骨断层及融合显像。([\s\S]*?)骨断层及CT融合显像：",
        re.DOTALL
    )
    m = pattern.search(text)
    if m:
        # 返回匹配内容的起止位置（不包括标记文本本身）
        return m.start(1), m.end(1)
    return None, None


# ----------------------------------------------------
# 🌟 将报告转成图片（逐字符高亮，并优化特定区域的段落间距）
# ----------------------------------------------------
def report_to_image(report_path, width=730):
    with open(report_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # 找到需要高亮的字符范围
    hl_start, hl_end = find_highlight_range(text)

    # 找到段落间距优化的分界点："骨断层及CT融合显像："
    NO_WRAP_MARKER = "骨断层及CT融合显像："
    # 优化从标记的结束位置开始
    optimized_spacing_start = text.find(NO_WRAP_MARKER)
    if optimized_spacing_start != -1:
        optimized_spacing_start += len(NO_WRAP_MARKER)
    else:
        # 如果没找到标记，则不进行间距优化
        optimized_spacing_start = len(text) + 1 

    # 计算图片高度
    line_height = FONT_SIZE + 6
    max_chars_per_line = 38
    
    # 预估行数，增加足够大的余量以避免黑屏
    estimated_lines = len(text) // max_chars_per_line + 50 
    img_height = line_height * estimated_lines

    img = Image.new("RGB", (width, img_height), "white")
    draw = ImageDraw.Draw(img)

    x, y = 10, 10
    char_idx = 0
    # 追踪上一个字符是否是换行符，用于在优化区域合并连续空行
    last_char_was_newline = True 

    for char in text:
        # 判断当前字符是否在高亮范围内
        in_highlight_range = (hl_start is not None and hl_start <= char_idx < hl_end)
        
        # 只有非空白字符才高亮
        is_highlight = in_highlight_range and char not in [' ', '\n', '\t', '\r']

        # 检查是否进入段落间距优化区域
        is_in_optimized_area = char_idx >= optimized_spacing_start

        # --- 换行处理逻辑 ---
        if char == '\n':
            if is_in_optimized_area:
                # 在优化区域，如果上一个字符已经是换行，则跳过此次换行（合并连续空行，减小段落间距）
                if last_char_was_newline:
                    char_idx += 1
                    continue
            
            # 正常换行 (或优化区内的第一个 \n)
            y += line_height
            x = 10
            last_char_was_newline = True
            char_idx += 1
            continue

        # 获取字符宽度
        bbox = draw.textbbox((0, 0), char, font=font)
        char_width = bbox[2] - bbox[0]
        
        # 自动换行逻辑：所有区域都保持自动换行（分段）
        if x + char_width > width - 10:
            y += line_height
            x = 10
        
        # 如果当前是非空白字符，重置标记
        if char not in [' ', '\t', '\r', '\n']:
             last_char_was_newline = False

        # 如果需要高亮，先画黄色背景
        if is_highlight:
            draw.rectangle(
                [(x - 2, y - 2), (x + char_width + 2, y + line_height - 4)],
                fill="#fff59d"
            )

        # 绘制字符
        draw.text((x, y), char, fill="black", font=font)
        x += char_width

        char_idx += 1

    # 裁剪到实际使用的高度
    actual_height = y + line_height + 10
    img = img.crop((0, 0, width, actual_height))

    return img


# ----------------------------------------------------
# 遍历患者图像并生成大图
# ----------------------------------------------------
patients = {}

for img_path in vis_dir.glob("*.png"):
    stem = img_path.stem
    base = stem.replace("_front", "").replace("_back", "")
    patients.setdefault(base, {})[stem.split("_")[-1]] = img_path

print(f"共检测到 {len(patients)} 个患者")


for pid, imgs in patients.items():
    print(f"\n处理患者：{pid}")

    front_img = imgs.get("front")
    back_img = imgs.get("back")

    if front_img is None or back_img is None:
        print(f"❌ {pid} 缺少 front/back 图，跳过")
        continue

    report_path = report_dir / f"{pid}.txt"
    if not report_path.exists():
        print(f"❌ 缺少报告：{report_path}")
        continue

    # 报告 → 图片（带字符级高亮和段落间距优化）
    report_img = report_to_image(report_path)

    # 打开 front/back
    try:
        img_f = Image.open(front_img).convert("RGB")
        img_b = Image.open(back_img).convert("RGB")
    except Exception as e:
        print(f"❌ {pid} 图像打开失败：{e}")
        continue

    # 对齐高度
    h = max(report_img.height, img_f.height, img_b.height)

    def pad(img, target_h):
        if img.height == target_h:
            return img
        new_img = Image.new("RGB", (img.width, target_h), "white")
        new_img.paste(img, (0, 0))
        return new_img

    report_img = pad(report_img, h)
    img_f = pad(img_f, h)
    img_b = pad(img_b, h)

    # 横向拼接
    total_w = report_img.width + img_f.width + img_b.width
    merged = Image.new("RGB", (total_w, h), "white")

    merged.paste(report_img, (0, 0))
    merged.paste(img_f, (report_img.width, 0))
    merged.paste(img_b, (report_img.width + img_f.width, 0))

    out_path = save_dir / f"{pid}_merged.png"
    merged.save(out_path)

    print(f"✅ 已生成：{out_path}")
