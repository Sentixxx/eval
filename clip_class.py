import torch
import clip
from PIL import Image
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cairosvg
import io
import gc
import contextlib
from functools import wraps
import time

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 文件限制上下文管理器，确保文件被正确关闭
@contextlib.contextmanager
def managed_open(filepath, mode='r'):
    try:
        f = open(filepath, mode)
        yield f
    finally:
        f.close()

# 添加重试装饰器处理暂时性错误
def retry_on_exception(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except (IOError, OSError) as e:
                    if "Too many open files" in str(e) and retries < max_retries - 1:
                        retries += 1
                        print(f"文件资源不足，等待 {delay} 秒后重试 ({retries}/{max_retries-1})...")
                        time.sleep(delay)
                        # 主动释放资源
                        gc.collect()
                    else:
                        raise
        return wrapper
    return decorator

class ClipClassifier:
    def __init__(self, model_name="ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.class_names = ["bicycle", "car", "motorcycle", "plane", "traffic light", "fire hydrant", "cat", "dog", "horse", "sheep","cow", "elephant", "zebra","giraffe"]
        self.text_prompts = [f"A sketch of a(n) {name}" for name in self.class_names]
        self.text_inputs = clip.tokenize(self.text_prompts).to(self.device)
        self.classification_results = {}
        self.confidence_scores = {}
        self.ground_truth_labels = {}  # 存储文件的真实标签
        self.accuracy = 0  # 存储整体准确率
        self.true_label = {
            "bicycle": [],
            "car": [],
            "motorcycle": [],
            "plane": [],
            "traffic light": [],
            "fire hydrant": [],
            "cat": [],
            "dog": [],
            "horse": [],
            "sheep": [],
            "cow": [],
            "elephant": [],
            "zebra": [],
            "giraffe": []
        }
        self.error_images = []
        self.batch_size = 16  # 批处理大小，减少同时打开的文件数量
        
    @retry_on_exception(max_retries=5, delay=2)
    def convert_svg_to_png(self, svg_path):
        """使用资源管理确保文件被关闭"""
        try:
            # 读取SVG文件内容，而不是直接传递URL
            with managed_open(svg_path, 'rb') as f:
                svg_data = f.read()
                
            # 将SVG转换为PNG
            png_data = cairosvg.svg2png(bytestring=svg_data, background_color="white")
            return Image.open(io.BytesIO(png_data))
        except Exception as e:
            print(f"SVG转换错误 {svg_path}: {e}")
            
            # 如果出现问题，尝试替代方法
            try:
                with managed_open(svg_path, 'rb') as f:
                    svg_data = f.read()
                    
                # 尝试不带背景色转换
                png_data = cairosvg.svg2png(bytestring=svg_data)
                svg_image = Image.open(io.BytesIO(png_data))
                
                # 创建白色背景
                white_bg = Image.new("RGB", svg_image.size, (255, 255, 255))
                
                if svg_image.mode == 'RGBA':
                    white_bg.paste(svg_image, (0, 0), svg_image.split()[3])
                else:
                    white_bg.paste(svg_image, (0, 0))
                
                return white_bg
            except Exception as e:
                print(f"无法处理SVG文件 {svg_path}: {e}")
                return None
    
    @retry_on_exception(max_retries=3, delay=1)
    def process_image(self, img_path, temp_dir=None, true_label=None):
        """处理单个图像，添加错误重试和资源管理"""
        try:
            if temp_dir:
                png_path = temp_dir / f"{img_path.stem}.png"
            
            # 根据图像类型处理
            if img_path.suffix.lower() == '.svg':
                image = self.convert_svg_to_png(img_path)
                if image is None:
                    return False
            else:
                # 使用上下文管理器打开图像
                with managed_open(img_path, 'rb') as f:
                    image = Image.open(io.BytesIO(f.read()))
                    # 立即复制图像数据，这样可以关闭文件
                    image = image.copy()
                
            # 处理透明通道
            if image.mode == 'RGBA':
                white_bg = Image.new("RGB", image.size, (255, 255, 255))
                white_bg.paste(image, (0, 0), image.split()[3])
                image = white_bg
            
            # 预处理并分类图像
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                text_features = self.model.encode_text(self.text_inputs)
                
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                
                values, indices = similarity[0].topk(3)
                
                top_class = self.class_names[indices[0].item()]
                self.classification_results[img_path.name] = top_class
                self.confidence_scores[img_path.name] = {
                    self.class_names[indices[i].item()]: values[i].item() for i in range(3)
                }
                
                # 如果提供了真实标签，则存储它
                if true_label:
                    self.ground_truth_labels[img_path.name] = true_label
                
                correct_mark = ""
                if true_label:
                    is_correct = top_class == true_label
                    correct_mark = "✓" if is_correct else "✗"
                    if is_correct:
                        self.true_label[true_label].append(str(img_path))
                    else:
                        self.error_images.append(str(img_path))
                
                print(f"图像: {img_path.name} | 预测类别: {top_class} | 置信度: {values[0].item():.2f} {correct_mark}")
                if true_label and not is_correct:
                    print(f"  真实类别: {true_label}")
                print(f"  次要预测: {self.class_names[indices[1].item()]} ({values[1].item():.2f}), {self.class_names[indices[2].item()]} ({values[2].item():.2f})")
                
            # 释放不需要的资源
            del image_input, image_features, text_features, similarity
            torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
            
            return True
        except Exception as e:
            print(f"处理图像 {img_path} 时出错: {e}")
            return False
    
    def classify_images(self, image_folder):
        """分类图像，使用批处理减少同时打开的文件数量"""
        # 获取所有图像文件
        image_files = []
        
        # 检查是否有子文件夹，如果有，则使用子文件夹名称作为标签
        folder_path = Path(image_folder)
        has_subfolders = any(item.is_dir() for item in folder_path.iterdir())
        
        if has_subfolders:
            print("检测到子文件夹结构，使用文件夹名称作为真实标签...")
            for subfolder in folder_path.iterdir():
                if subfolder.is_dir():
                    label = subfolder.name
                    # 检查标签是否在我们的类别列表中
                    if label not in self.class_names:
                        print(f"警告: 子文件夹 '{label}' 不在预设类别列表中")
                    
                    for ext in ['*.svg', '*.png', '*.jpg', '*.jpeg']:
                        image_files.extend([(img_path, label) for img_path in subfolder.glob(ext)])
        else:
            # 没有子文件夹，按照原来的方式处理
            for ext in ['*.svg', '*.png', '*.jpg', '*.jpeg']:
                image_files.extend([(img_path, None) for img_path in folder_path.glob(ext)])
        
        if not image_files:
            print(f"在 {image_folder} 中未找到支持的图像文件")
            return 0
            
        # 创建临时目录
        temp_dir = Path("temp_png")
        temp_dir.mkdir(exist_ok=True)
        
        # 批量处理图像以减少同时打开的文件数
        total_processed = 0
        batch_size = self.batch_size
        
        for i in range(0, len(image_files), batch_size):
            batch = image_files[i:i+batch_size]
            print(f"处理批次 {i//batch_size + 1}/{(len(image_files)-1)//batch_size + 1} ({len(batch)} 个文件)")
            
            for img_path, true_label in batch:
                success = self.process_image(img_path, temp_dir, true_label)
                if success:
                    total_processed += 1
                
                # 主动执行垃圾回收释放资源
                if total_processed % 10 == 0:
                    gc.collect()
        
        # 清理临时文件
        for temp_file in temp_dir.glob('*'):
            try:
                os.remove(temp_file)
            except Exception as e:
                print(f"无法删除临时文件 {temp_file}: {e}")
        
        try:
            os.rmdir(temp_dir)
        except Exception as e:
            print(f"无法删除临时目录 {temp_dir}: {e}")
            
        return total_processed
    
    def calculate_accuracy(self):
        """计算分类准确率"""
        if not self.ground_truth_labels:
            print("没有提供真实标签，无法计算准确率")
            return 0
            
        correct = 0
        total = 0
        
        # 按类别计算准确率
        class_accuracy = {}
        class_counts = {}
        
        for filename, true_label in self.ground_truth_labels.items():
            predicted_label = self.classification_results.get(filename)
            if predicted_label:
                total += 1
                if predicted_label == true_label:
                    correct += 1
                
                # 按类别统计
                if true_label not in class_counts:
                    class_counts[true_label] = 0
                    class_accuracy[true_label] = 0
                
                class_counts[true_label] += 1
                if predicted_label == true_label:
                    class_accuracy[true_label] += 1
        
        # 计算总体准确率
        self.accuracy = correct / total if total > 0 else 0
        print(f"\n总体准确率: {self.accuracy:.2%} ({correct}/{total})")
        
        # 按类别显示准确率
        print("\n各类别准确率:")
        for label, count in class_counts.items():
            acc = class_accuracy[label] / count if count > 0 else 0
            print(f"类别: {label} | 准确率: {acc:.2%} ({class_accuracy[label]}/{count})")
            
        return self.accuracy
    
    def save_error_images(self):
        if self.error_images:
            txt_path = Path("error_images.txt")
            with managed_open(txt_path, "w") as f:
                for img_path in self.error_images:
                    f.write(f"{img_path}\n")
    
    def generate_statistics(self, total_images):
        # 生成统计结果
        class_counts = {}
        for class_name in self.classification_results.values():
            if class_name in class_counts:
                class_counts[class_name] += 1
            else:
                class_counts[class_name] = 1
        
        print("\n分类统计结果:")
        for class_name, count in class_counts.items():
            print(f"类别: {class_name} | 数量: {count} | 比例: {count / total_images:.2%}")
            
        return class_counts
    
    @retry_on_exception(max_retries=3, delay=2)
    def visualize_results(self, class_counts, total_images):
        # 创建可视化前清理资源
        plt.close('all')
        gc.collect()
        
        try:
            # 创建可视化
            classes = list(class_counts.keys())
            counts = list(class_counts.values())
            percentages = [count / total_images * 100 for count in counts]
            
            # 条形图
            plt.figure(figsize=(12, 6))
            bars = plt.bar(classes, counts, color='skyblue')
            plt.xlabel('类别')
            plt.ylabel('图像数量')
            plt.title('各类别图像分布')
            plt.xticks(rotation=45, ha='right')
            
            # 添加数值标签
            for i, bar in enumerate(bars):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                         f"{counts[i]} ({percentages[i]:.1f}%)",
                         ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plt.savefig("class_distribution.png")
            plt.close()
            
            # 饼图
            plt.figure(figsize=(10, 10))
            plt.pie(counts, labels=classes, autopct='%1.1f%%', startangle=90, 
                    shadow=True, wedgeprops={'edgecolor': 'w'})
            plt.title('各类别图像比例')
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig("class_pie.png")
            plt.close()
            
            # 如果有准确率数据，绘制准确率图
            if hasattr(self, 'accuracy') and self.ground_truth_labels:
                class_accuracy = {}
                class_counts = {}
                
                for filename, true_label in self.ground_truth_labels.items():
                    predicted_label = self.classification_results.get(filename)
                    if predicted_label:
                        if true_label not in class_counts:
                            class_counts[true_label] = 0
                            class_accuracy[true_label] = 0
                        
                        class_counts[true_label] += 1
                        if predicted_label == true_label:
                            class_accuracy[true_label] += 1
                
                # 计算每个类别的准确率
                acc_labels = []
                acc_values = []
                
                for label, count in class_counts.items():
                    if count > 0:
                        acc = class_accuracy[label] / count
                        acc_labels.append(label)
                        acc_values.append(acc * 100)  # 转换为百分比
                
                if acc_labels:
                    plt.figure(figsize=(12, 6))
                    bars = plt.bar(acc_labels, acc_values, color='lightgreen')
                    plt.xlabel('类别')
                    plt.ylabel('准确率 (%)')
                    plt.title('各类别分类准确率')
                    plt.xticks(rotation=45, ha='right')
                    plt.ylim(0, 100)
                    
                    # 添加数值标签
                    for i, bar in enumerate(bars):
                        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                                f"{acc_values[i]:.1f}%",
                                ha='center', va='bottom', fontsize=9)
                    
                    plt.tight_layout()
                    plt.savefig("class_accuracy.png")
                    plt.close()
            
            print("\n可视化图表已保存为PNG文件")
        
        except Exception as e:
            print(f"创建可视化时出错: {e}")
    
    def visualize_confusion_matrix(self):
        """生成混淆矩阵的可视化"""
        if not self.ground_truth_labels:
            print("没有提供真实标签，无法创建混淆矩阵")
            return
        
        # 清理资源
        plt.close('all')
        gc.collect()
        
        try:
            # 获取所有出现在标签中的类别
            all_classes = sorted(list(set(self.ground_truth_labels.values())))
            n_classes = len(all_classes)
            
            # 创建混淆矩阵
            confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
            class_to_idx = {cls: i for i, cls in enumerate(all_classes)}
            
            for filename, true_label in self.ground_truth_labels.items():
                predicted_label = self.classification_results.get(filename)
                if predicted_label:
                    true_idx = class_to_idx[true_label]
                    pred_idx = class_to_idx.get(predicted_label, -1)
                    if pred_idx >= 0:
                        confusion_matrix[true_idx, pred_idx] += 1
            
            # 绘制混淆矩阵
            plt.figure(figsize=(10, 8))
            plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('混淆矩阵')
            plt.colorbar()
            
            # 设置轴标签
            tick_marks = np.arange(n_classes)
            plt.xticks(tick_marks, all_classes, rotation=45, ha='right')
            plt.yticks(tick_marks, all_classes)
            
            # 添加文本标注
            thresh = confusion_matrix.max() / 2.0
            for i in range(n_classes):
                for j in range(n_classes):
                    plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                             ha="center", va="center",
                             color="white" if confusion_matrix[i, j] > thresh else "black")
            
            plt.tight_layout()
            plt.ylabel('真实标签')
            plt.xlabel('预测标签')
            plt.savefig("confusion_matrix.png")
            plt.close()
            
            print("\n混淆矩阵已保存为 confusion_matrix.png")
            
        except Exception as e:
            print(f"创建混淆矩阵时出错: {e}")


def main():
    # 初始化分类器
    classifier = ClipClassifier()
    
    # 设置图像文件夹路径
    image_folder = input("请输入图像文件夹路径 (默认为'./images'): ") or "./images"
    image_folder = Path(image_folder)
    
    # 确保文件夹存在
    if not image_folder.exists():
        print(f"文件夹 {image_folder} 不存在，是否创建? (y/n)")
        if input().lower() == 'y':
            image_folder.mkdir(parents=True, exist_ok=True)
        else:
            print("程序已退出")
            return
    
    # 分类图像
    total_images = classifier.classify_images(image_folder)
    
    # 如果没有图像则退出
    if total_images == 0:
        print("未找到图像，程序退出")
        return
    
    # 计算准确率（如果有真实标签）
    if classifier.ground_truth_labels:
        classifier.calculate_accuracy()
        classifier.visualize_confusion_matrix()
    
    # 生成统计数据
    class_counts = classifier.generate_statistics(total_images)
    
    # 可视化结果
    classifier.visualize_results(class_counts, total_images)
    
    # 保存错误分类的图像列表
    if classifier.error_images:
        classifier.save_error_images()
        print(f"已将 {len(classifier.error_images)} 个错误分类的图像路径保存到 error_images.txt")


if __name__ == "__main__":
    main()