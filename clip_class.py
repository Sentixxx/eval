import torch
import clip
from PIL import Image
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cairosvg
import io


class ClipClassifier:
    def __init__(self, model_name="ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.class_names = ["cat", "dog", "horse", "person", "car", "airplane", "chair", "bottle", 
                           "watch", "tree", "camel", "rose", "sunflower", "elephant", "giraffe", 
                           "kangaroo", "panda", "penguin", "tiger", "zebra"]
        self.text_prompts = [f"A sketch of a(n) {name}" for name in self.class_names]
        self.text_inputs = clip.tokenize(self.text_prompts).to(self.device)
        self.classification_results = {}
        self.confidence_scores = {}
        
    def convert_svg_to_png(self, svg_path):
        try:
            png_data = cairosvg.svg2png(url=str(svg_path), background_color="white")
            return Image.open(io.BytesIO(png_data))
        except Exception as e:
            print(f"SVG转换错误 {svg_path}: {e}")
            return None
    
    def process_image(self, img_path, temp_dir=None):
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
                
                print(f"图像: {img_path.name} | 预测类别: {top_class} | 置信度: {values[0].item():.2f}")
                print(f"  次要预测: {self.class_names[indices[1].item()]} ({values[1].item():.2f}), {self.class_names[indices[2].item()]} ({values[2].item():.2f})")
                
            return True
        except Exception as e:
            print(f"处理图像 {img_path} 时出错: {e}")
            return False
    
    def classify_images(self, image_folder):
        # 获取所有图像文件
        image_files = []
        for ext in ['*.svg', '*.png', '*.jpg', '*.jpeg']:
            image_files.extend(list(Path(image_folder).glob(ext)))
        
        if not image_files:
            print(f"在 {image_folder} 中未找到支持的图像文件")
            return
            
        # 创建临时目录
        temp_dir = Path("temp_png")
        temp_dir.mkdir(exist_ok=True)
        
        # 处理所有图像
        for img_path in image_files:
            self.process_image(img_path, temp_dir)
        
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
                     f'{height}个 ({percentage:.1f}%)',
                     ha='center', va='bottom', fontsize=9)
        
        plt.title('图像分类结果统计')
        plt.xlabel('类别')
        plt.ylabel('图像数量')
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
        plt.title('各类别图像占比')
        plt.tight_layout()
        
        plt.savefig('classification_pie_chart.png', dpi=300)
        plt.close()
        
        print(f"\n可视化结果已保存为 'classification_results.png' 和 'classification_pie_chart.png'")


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
    
    # 生成统计数据
    class_counts = classifier.generate_statistics(total_images)
    
    # 可视化结果
    classifier.visualize_results(class_counts, total_images)


if __name__ == "__main__":
    main()