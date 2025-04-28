from dreamsim import dreamsim
from PIL import Image
import os
import clip_class
import torch
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse
from functools import lru_cache
import cv2

# 设置设备，优先使用GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = dreamsim(pretrained=True, device=device)

# 缓存预处理过的图像，减少重复计算
@lru_cache(maxsize=100)
def preprocess_image(img_path):
    try:
        # 检查文件是否存在
        if not os.path.exists(img_path):
            print(f"图像不存在: {img_path}")
            return None
            
        # 检查文件扩展名
        ext = os.path.splitext(img_path)[1].lower()
        if ext == '.svg':
            # 使用clip_class中的转换方法处理SVG文件
            try:
                # 初始化分类器（如果尚未初始化）
                if not hasattr(preprocess_image, 'classifier'):
                    preprocess_image.classifier = clip_class.ClipClassifier()
                
                # 转换SVG为PNG
                img = preprocess_image.classifier.convert_svg_to_png(img_path)
                if img is not None:
                    return preprocess(img).to(device)
                else:
                    print(f"无法转换SVG文件: {img_path}")
                    return None
            except Exception as svg_e:
                print(f"SVG转换错误: {img_path}: {svg_e}")
                return None
        
        # 使用cv2读取和预处理图像
        target_size = (224, 224)
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}")
            return None
            
        # 计算缩放比例，保持纵横比
        h, w = img.shape[:2]
        ratio = min(target_size[0] / w, target_size[1] / h)
        new_size = (int(w * ratio), int(h * ratio))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
            
        # 创建白色画布并居中放置图像
        canvas = np.ones((target_size[1], target_size[0], 3), dtype=np.uint8) * 255
        
        x_offset = (target_size[0] - w) // 2
        y_offset = (target_size[1] - h) // 2
        
        canvas[y_offset:y_offset+h, x_offset:x_offset+w] = img
        
        # 转换为PIL图像
        img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        return preprocess(pil_img).to(device)
    except Exception as e:
        print(f"图像预处理错误 {img_path}: {e}")
        return None

def read_origin_images(path, allowed_extensions=('.png', '.jpg', '.jpeg', '.svg')):
    """读取原始图像路径，不加载到内存中"""
    labels = {
        "bicycle": {}, "car": {}, "motorcycle": {}, "plane": {},
        "traffic light": {}, "fire hydrant": {}, "cat": {}, "dog": {},
        "horse": {}, "sheep": {}, "cow": {}, "elephant": {},
        "zebra": {}, "giraffe": {}
    }
    
    path = Path(path)
    if not path.exists():
        print(f"警告: 路径 {path} 不存在")
        return labels
    
    for label in tqdm(labels, desc="读取原始图像路径"):
        label_path = path / label
        if not label_path.exists():
            print(f"警告: 标签路径 {label_path} 不存在")
            continue
            
        for file in os.listdir(label_path):
            # 只处理允许的文件扩展名
            if file.lower().endswith(allowed_extensions):
                file_name = file.split(".")[0]
                img_path = label_path / file
                if img_path.exists():
                    labels[label][file_name] = str(img_path)
                else:
                    print(f"警告: 图像 {img_path} 不存在")
    
    return labels

def calculate_similarity_batch(img_list1, img_list2, batch_size=16):
    """批量计算相似度，提高处理效率"""
    results = []
    
    for i in range(0, len(img_list1), batch_size):
        batch_img1 = img_list1[i:i+batch_size]
        batch_img2 = img_list2[i:i+batch_size]
        
        # 确保两个批次长度一致
        min_len = min(len(batch_img1), len(batch_img2))
        batch_img1 = batch_img1[:min_len]
        batch_img2 = batch_img2[:min_len]
        
        # 创建批处理张量
        batch_tensor1 = []
        batch_tensor2 = []
        
        for j in range(len(batch_img1)):
            # 处理第一个列表中的图像
            if isinstance(batch_img1[j], str):
                processed_img1 = preprocess_image(batch_img1[j])
            else:
                processed_img1 = preprocess(batch_img1[j]).to(device)
                
            # 处理第二个列表中的图像
            if isinstance(batch_img2[j], str):
                processed_img2 = preprocess_image(batch_img2[j])
            else:
                processed_img2 = preprocess(batch_img2[j]).to(device)
                
            if processed_img1 is not None and processed_img2 is not None:
                batch_tensor1.append(processed_img1.unsqueeze(0))
                batch_tensor2.append(processed_img2.unsqueeze(0))
        
        if not batch_tensor1 or not batch_tensor2:
            continue
            
        # 合并批处理张量
        batch_tensor1 = torch.cat(batch_tensor1).to(device)
        batch_tensor2 = torch.cat(batch_tensor2).to(device)
        
        # 批量计算距离
        with torch.no_grad():
            try:
                distances = model(batch_tensor1, batch_tensor2)
                results.extend(distances.cpu().numpy())
            except Exception as e:
                print(f"批处理计算错误: {e}")
                # 回退到单个图像计算
                for j in range(len(batch_tensor1)):
                    try:
                        distance = model(batch_tensor1[j].unsqueeze(0), batch_tensor2[j].unsqueeze(0))
                        results.append(distance.item())
                    except Exception as inner_e:
                        print(f"单图像计算错误: {inner_e}")
                        results.append(float('inf'))
    
    return results

def calculate_similarity(img1, img2):
    """计算单对图像的相似度"""
    try:
        # 检查图像是否已经是预处理后的张量或路径
        if isinstance(img1, str):
            img1 = preprocess_image(img1)
        elif not isinstance(img1, torch.Tensor):
            img1 = preprocess(img1).to(device)
            
        if isinstance(img2, str):
            img2 = preprocess_image(img2)
        elif not isinstance(img2, torch.Tensor):
            img2 = preprocess(img2).to(device)
            
        if img1 is None or img2 is None:
            return float('inf')
            
        with torch.no_grad():
            distance = model(img1.unsqueeze(0) if img1.dim() == 3 else img1, 
                             img2.unsqueeze(0) if img2.dim() == 3 else img2)
        return distance.item() if distance.numel() == 1 else distance
    except Exception as e:
        print(f"计算相似度时出错: {e}")
        return float('inf')  # 返回无限大表示计算失败

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='图像相似度计算与分类')
    parser.add_argument('--image_folder', type=str, default='./images',
                        help='图像文件夹路径')
    parser.add_argument('--origin_folder', type=str, default='./origin_images',
                        help='原始图像文件夹路径')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批处理大小')
    parser.add_argument('--no_prompt', action='store_true',
                        help='不显示交互式提示，直接使用命令行参数')
    parser.add_argument('--output_file', type=str, default='',
                        help='将结果输出到指定文件')
    parser.add_argument('--skip_visualization', action='store_true',
                        help='跳过结果可视化')
    parser.add_argument('--allowed_extensions', type=str, default='.png,.jpg,.jpeg,.svg',
                        help='允许的文件扩展名，用逗号分隔')
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='相似度阈值，低于此值的结果将被过滤')
    return parser.parse_args()

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_arguments()
    
    # 初始化分类器
    classifier = clip_class.ClipClassifier()
    
    # 处理允许的文件扩展名
    allowed_extensions = tuple(args.allowed_extensions.split(','))
    
    # 设置图像文件夹路径
    if args.no_prompt:
        image_folder = Path(args.image_folder)
    else:
        image_folder = input(f"请输入图像文件夹路径 (默认为'{args.image_folder}'): ") or args.image_folder
        image_folder = Path(image_folder)
    
    # 确保文件夹存在
    if not image_folder.exists():
        if args.no_prompt:
            print(f"文件夹 {image_folder} 不存在，将自动创建")
            image_folder.mkdir(parents=True, exist_ok=True)
        else:
            print(f"文件夹 {image_folder} 不存在，是否创建? (y/n)")
            if input().lower() == 'y':
                image_folder.mkdir(parents=True, exist_ok=True)
            else:
                print("程序已退出")
                exit()
    
    # 读取原始图像
    if args.no_prompt:
        origin_folder = args.origin_folder
    else:
        origin_folder = input(f"请输入原始图像文件夹路径 (默认为'{args.origin_folder}'): ") or args.origin_folder
    
    print(f"正在读取原始图像从 {origin_folder}...")
    origin_images = read_origin_images(origin_folder, allowed_extensions)
    
    # 分类图像
    print("开始分类图像...")
    total_images = classifier.classify_images(image_folder)
    
    # 如果没有找到图像则退出
    if not total_images:
        print("没有找到图像，程序退出")
        exit()
    
    # 计算准确率（如果有真实标签）
    if classifier.ground_truth_labels:
        classifier.calculate_accuracy()
    
    # 生成统计数据
    class_counts = classifier.generate_statistics(total_images)
    
    # 可视化结果
    if not args.skip_visualization:
        classifier.visualize_results(class_counts, total_images)

    # 计算相似度
    print("\n计算每个类别的相似度:")
    similarity_results = {}
    
    for label in tqdm(classifier.true_label, desc="处理类别"):
        # 跳过空类别
        if not classifier.true_label[label]:
            continue
            
        img_batch_origin = []
        img_batch_sketch = []
        paths = []
        
        # 准备批处理数据
        for img_path in classifier.true_label[label]:
            img_path = Path(img_path)
            img_name = img_path.name
            
            # 获取文件名，无论文件扩展名是什么
            true_name = img_name.split("_")[0]
            
            if true_name not in origin_images.get(label, {}):
                print(f"警告: 找不到原始图像 {true_name} 在类别 {label} 中")
                continue
                
            origin_img_path = origin_images[label][true_name]
            sketch_path = str(image_folder / label / img_name)
            
            # 验证文件存在
            if not os.path.exists(sketch_path):
                print(f"警告: 草图文件不存在: {sketch_path}")
                continue
                
            img_batch_origin.append(origin_img_path)
            img_batch_sketch.append(sketch_path)
            paths.append(img_name)
        
        # 如果没有可处理的图像对，则跳过
        if not img_batch_origin:
            continue
            
        # 使用单张处理或批处理，取决于图像数量
        if len(img_batch_origin) == 1:
            similarity = calculate_similarity(img_batch_origin[0], img_batch_sketch[0])
            similarities = [similarity]
        else:
            try:
                similarities = calculate_similarity_batch(img_batch_origin, img_batch_sketch, args.batch_size)
            except Exception as e:
                print(f"批处理计算失败，切换到单图像处理: {e}")
                similarities = []
                for i in range(len(img_batch_origin)):
                    try:
                        similarity = calculate_similarity(img_batch_origin[i], img_batch_sketch[i])
                        similarities.append(similarity)
                    except Exception as single_e:
                        print(f"单图像处理失败 ({img_batch_origin[i]} -> {img_batch_sketch[i]}): {single_e}")
                        similarities.append(float('inf'))
        
        # 检查和过滤无效结果
        if not similarities or all(s == float('inf') for s in similarities):
            print(f"类别 {label} 中的所有相似度计算均失败")
            continue
        
        # 计算平均相似度
        valid_similarities = [s for s in similarities if s != float('inf') and s >= args.threshold]
        if valid_similarities:
            avg_similarity = sum(valid_similarities) / len(valid_similarities)
            similarity_results[label] = {
                'average': avg_similarity,
                'count': len(valid_similarities),
                'individual': dict(zip([p for i, p in enumerate(paths) if similarities[i] != float('inf') and similarities[i] >= args.threshold], valid_similarities))
            }
            print(f"类别: {label} | 平均相似度: {avg_similarity:.4f} | 样本数: {len(valid_similarities)}")
        else:
            print(f"类别: {label} | 没有有效的相似度结果")
    
    # 输出总体结果
    if similarity_results:
        total_avg = sum(r['average'] for r in similarity_results.values()) / len(similarity_results)
        print(f"\n总体平均相似度: {total_avg:.4f}")
        
        # 按相似度排序
        sorted_results = sorted(similarity_results.items(), key=lambda x: x[1]['average'])
        print("\n各类别相似度排名:")
        for i, (label, data) in enumerate(sorted_results, 1):
            print(f"{i}. {label}: {data['average']:.4f} ({data['count']} 样本)")
        
        # 如果指定了输出文件，将结果写入文件
        if args.output_file:
            import json
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'total_average': total_avg,
                    'results': {k: {'average': v['average'], 'count': v['count']} 
                              for k, v in similarity_results.items()}
                }, f, ensure_ascii=False, indent=4)
            print(f"\n结果已保存到: {args.output_file}")
    else:
        print("\n没有计算出有效的相似度结果")

    


