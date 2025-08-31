import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import warnings
warnings.filterwarnings('ignore')
import shutil
from torch.nn import functional as F
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#定义特征提取器 cnn
class StyleFeatureExtractor:
    def __init__(self):
        self.model = models.resnet50(pretrained=True)
        self.model = self.model.to(device)

        # 选择要提取特征的中间层（这里以ResNet的layer3为例）
        self.target_layer = self.model.layer3
        self.features = None  # 用于存储中间特征

        # 注册钩子（Hook）来获取中间层的输出
        def get_features(module, input, output):
            self.features = output

        self.hook = self.target_layer.register_forward_hook(get_features)

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, image_paths, batch_size=2):
        dataset = ImagePathDataset(image_paths, self.transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False) #shuffle必须设置为False

        all_features = []
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(device)
                _ = self.model(batch)  # 前向传播，钩子会自动捕获中间特征

                # 对获取到的特征图进行全局平均池化
                # self.features的形状是 [batch_size, channels, height, width]
                batch_features = F.adaptive_avg_pool2d(self.features, (1, 1))
                batch_features = batch_features.view(batch.size(0), -1)  # 展平

                all_features.append(batch_features.cpu().numpy())

        # 移除钩子
        self.hook.remove()
        return np.vstack(all_features)

class ImagePathDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

def evaluate_clustering(features, cluster_labels, output_dir="clustering_evaluation"):
    """
   聚类效果评估函数

    参数:
    features: 用于聚类的特征矩阵 (n_samples, n_features)
    cluster_labels: 聚类标签数组 (n_samples,)
    output_dir: 输出目录，用于保存评估结果
    """

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 1. 计算内部评估指标
    silhouette_avg = silhouette_score(features, cluster_labels)
    calinski_harabasz = calinski_harabasz_score(features, cluster_labels)
    davies_bouldin = davies_bouldin_score(features, cluster_labels)

    # 2. 可视化聚类分布
    plt.figure(figsize=(8, 5))
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    bars = plt.bar(cluster_counts.index.astype(str), cluster_counts.values)

    # 在柱状图上添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}', ha='center', va='bottom')

    plt.xlabel('Cluster')
    plt.ylabel('Number of Images')
    plt.title('Distribution of Images Across Clusters')
    plt.savefig(os.path.join(output_dir, 'cluster_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 生成评估报告
    n_clusters = len(np.unique(cluster_labels))

    report = f"""
聚类效果评估报告
======================

数据集信息:
- 总图像数: {len(cluster_labels)}
- 聚类数量: {n_clusters}

评估指标:
- 轮廓系数: {silhouette_avg:.4f}
- Calinski-Harabasz指数: {calinski_harabasz:.4f}
- Davies-Bouldin指数: {davies_bouldin:.4f}

聚类分布:
{cluster_counts.to_string()}

指标解释:
- 轮廓系数: [-1, 1]，越接近1越好
- Calinski-Harabasz指数: 越高越好
- Davies-Bouldin指数: 越低越好
"""

    # 保存报告到文件
    with open(os.path.join(output_dir, 'evaluation_report.txt'), 'w') as f:
        f.write(report)

    # 打印报告到控制台
    print(report)

    # 返回评估结果
    return {
        'silhouette_score': silhouette_avg,
        'calinski_harabasz_score': calinski_harabasz,
        'davies_bouldin_score': davies_bouldin,
        'cluster_distribution': cluster_counts.to_dict(),
        'n_clusters': n_clusters
    }

def tsne_visualize(features,labels,title:str):
    print('正在进行t-SNE可视化...')
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels,
                          cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(title)
    plt.savefig(title+'.png')
    plt.show()

#3. 加载数据并进行特征聚类
def main():
    root_dir = r'D:\dev\python\Datasets\Fundus-doFE\Fundus'
    image_paths = []
    for i in range(1,5):
        data_dir = os.path.join(root_dir, 'Domain'+str(i),'train\ROIs\image')
        image_paths.extend([os.path.join(data_dir,f) for f in os.listdir(data_dir) if f.endswith(('.jpg','.jpeg','.png'))])
        data_dir.replace('train','test')
        image_paths.extend([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    print(f'找到{len(image_paths)}张图像')
    domain_labels = [int(path.split(os.sep)[-5][-1]) for path in image_paths]

    extractor = StyleFeatureExtractor()
    print('正在提取特征...')
    features = extractor.extract_features(image_paths)
    print(f'特征形状:{features.shape}')

    #做PCA降维
    pca = PCA(n_components=50,random_state=42)
    features_reduced = pca.fit_transform(features)
    print(f'降维后特征形状：{features_reduced.shape}')

    tsne_visualize(features_reduced,domain_labels,title = 'Origin Domain Distribution')

    #做K-Means聚类
    n_clusters = 6
    kmeans = KMeans(n_clusters=n_clusters, random_state=42,n_init = 10)
    cluster_labels = kmeans.fit_predict(features_reduced)

    tsne_visualize(features_reduced,cluster_labels,title = 'Visualization Of Clusters')

    #分析每个簇的图像
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_images = [image_paths[i] for i in cluster_indices[:5]] #每个簇取五张示例

        print(f"\n簇 {cluster_id} 包含 {len(cluster_indices)} 张图像")
        print("示例图像:")
        for img_path in cluster_images:
            print(f"  - {os.path.basename(img_path)}")

    #保存聚类结果
    results = pd.DataFrame({
        'image_path':image_paths,
        'cluster_label':cluster_labels
    })
    results.to_csv('cluster_labels.csv',index=False)
    print('\n聚类结果已保存到 cluster_labels.csv')

    #评估聚类效果
    evaluation_results = evaluate_clustering(
        features_reduced,  # 降维后的特征
        cluster_labels,  # 聚类标签
        output_dir="./"  # 输出目录
    )


    #重新划分数据集
    if input('Redivide the dataset?').lower() == 'y':
        save_base_dir = r'D:\dev\python\Datasets\Fundus_cluster'
        for i in range(len(image_paths)):
            print(f'正在复制来自{image_paths[i]}的图像...')
            save_path = os.path.join(save_base_dir,'domain'+str(cluster_labels[i]+1))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            shutil.copy(image_paths[i], save_path)
            print(f"成功复制到{save_path}")


if __name__ == '__main__':
    main()