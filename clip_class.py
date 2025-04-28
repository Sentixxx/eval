import torch
import clip
from PIL import Image
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cairosvg
import io

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

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
        
    def convert_svg_to_png(self, svg_path):
        try:
            png_data = cairosvg.svg2png(url=str(svg_path), background_color="white")
            return Image.open(io.BytesIO(png_data))
        except Exception as e:
            print(f"SVG转换错误 {svg_path}: {e}")
            return None
    
    def process_image(self, img_path, temp_dir=None, true_label=None):
        try:
            if temp_dir:
                png_path = temp_dir / f"{img_path.stem}.png"
            
            if img_path.suffix.lower() == '.svg':
                try:
                    png_data = cairosvg.svg2png(url=str(img_path), background_color="white")
                    image = Image.open(io.BytesIO(png_data))
                except Exception as svg_error:
                    print(f"SVG转换错误 {img_path}: {svg_error}")
                    try:
                        png_data = cairosvg.svg2png(url=str(img_path))
                        svg_image = Image.open(io.BytesIO(png_data))
                        
                        white_bg = Image.new("RGB", svg_image.size, (255, 255, 255))
                        
                        if svg_image.mode == 'RGBA':
                            white_bg.paste(svg_image, (0, 0), svg_image.split()[3])
                        else:
                            white_bg.paste(svg_image, (0, 0))
                        
                        image = white_bg
                    except Exception as e:
                        print(f"无法处理SVG文件 {img_path}: {e}")
                        return False
            else:
                image = Image.open(img_path)
                
            if image.mode == 'RGBA':
                white_bg = Image.new("RGB", image.size, (255, 255, 255))
                white_bg.paste(image, (0, 0), image.split()[3])
                image = white_bg
            
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
                
                print(f"图像: {img_path.name} | 预测类别: {top_class} | 置信度: {values[0].item():.2f} {correct_mark}")
                if true_label and not is_correct:
                    print(f"  真实类别: {true_label}")
                print(f"  次要预测: {self.class_names[indices[1].item()]} ({values[1].item():.2f}), {self.class_names[indices[2].item()]} ({values[2].item():.2f})")
                
            return True
        except Exception as e:
            print(f"处理图像 {img_path} 时出错: {e}")
            return False
    
    def classify_images(self, image_folder):
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
                        for img_path in subfolder.glob(ext):
                            image_files.append((img_path, label))
        else:
            # 没有子文件夹，按照原来的方式处理
            for ext in ['*.svg', '*.png', '*.jpg', '*.jpeg']:
                for img_path in folder_path.glob(ext):
                    image_files.append((img_path, None))
        
        if not image_files:
            print(f"在 {image_folder} 中未找到支持的图像文件")
            return 0
            
        # 创建临时目录
        temp_dir = Path("temp_png")
        temp_dir.mkdir(exist_ok=True)
        
        # 处理所有图像
        for img_path, true_label in image_files:
            self.process_image(img_path, temp_dir, true_label)
        
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
            
        return len(image_files)
    
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
    
    def visualize_results(self, class_counts, total_images):
        # 创建可视化
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        percentages = [count / total_images * 100 for count in counts]
        
        # 条形图
        plt.figure(figsize=(12, 6))
        bars = plt.bar(classes, counts, color='skyblue')
        
        for bar, percentage in zip(bars, percentages):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                     f'{height} ({percentage:.1f}%)',
                     ha='center', va='bottom', fontsize=9)
        
        plt.title('Classification Results Statistics')
        plt.xlabel('Category')
        plt.ylabel('Image Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plt.savefig('classification_results.png')
        plt.close()
        
        # 饼图
        plt.figure(figsize=(12, 10))
        plt.pie(counts, labels=classes, autopct='%1.1f%%', startangle=90, shadow=True, 
                labeldistance=1.1,
                pctdistance=0.85,   
                textprops={'fontsize': 10})
        
        plt.legend(classes, loc="best", bbox_to_anchor=(0.9, 0.1, 0.5, 0.5))
        
        plt.axis('equal')
        plt.title('Image Category Distribution')
        plt.tight_layout()
        
        plt.savefig('classification_pie_chart.png', dpi=300)
        plt.close()
        
        # 如果有真实标签，则绘制混淆矩阵
        if self.ground_truth_labels:
            self.visualize_confusion_matrix()
        
        print(f"\n可视化结果已保存为 'classification_results.png' 和 'classification_pie_chart.png'")
    
    def visualize_confusion_matrix(self):
        """绘制混淆矩阵可视化"""
        from sklearn.metrics import confusion_matrix
        import pandas as pd
        import seaborn as sns
        
        # 获取所有唯一的类别标签
        all_classes = sorted(list(set(self.ground_truth_labels.values())))
        
        # 准备真实标签和预测标签的列表
        y_true = []
        y_pred = []
        
        for filename, true_label in self.ground_truth_labels.items():
            pred_label = self.classification_results.get(filename)
            if pred_label:
                y_true.append(true_label)
                y_pred.append(pred_label)
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred, labels=all_classes)
        
        # 创建DataFrame以便更好地展示
        cm_df = pd.DataFrame(cm, index=all_classes, columns=all_classes)
        
        # 绘制热力图
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        plt.savefig('confusion_matrix.png', dpi=300)
        plt.close()
        
        print("混淆矩阵已保存为 'confusion_matrix.png'")


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
    
    # 如果没有找到图像则退出
    if not total_images:
        return
    
    # 计算准确率（如果有真实标签）
    if classifier.ground_truth_labels:
        classifier.calculate_accuracy()
    
    # 生成统计数据
    class_counts = classifier.generate_statistics(total_images)
    
    # 可视化结果
    classifier.visualize_results(class_counts, total_images)


if __name__ == "__main__":
    main()