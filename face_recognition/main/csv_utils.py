import torch
import pandas as pd
import os
from pypinyin import lazy_pinyin
class FaceDatabase:
    def __init__(self, csv_path='feature/face_meta.csv'):
        """
        初始化人脸数据库
        参数:
            csv_path: 存储人脸元数据的CSV文件路径，默认为'face_meta.csv'
        """
        self.csv_path = csv_path  # 元数据文件路径
        self.df = self._init_db()  # 初始化数据库DataFrame
        self.last_similarity = 0  # 记录最后一次比对的相似度分数
        
    def _init_db(self):
        """
        内部方法：初始化数据库
        尝试从CSV文件加载数据，如果文件不存在则创建空DataFrame
        返回:
            pandas.DataFrame: 包含name和feature_path两列的DataFrame
        """
        try:
            return pd.read_csv(self.csv_path)  # 尝试读取CSV文件
        except:
            # 如果文件不存在，创建包含两列的空DataFrame
            return pd.DataFrame(columns=['name', 'feature_path'])

    def add_face(self, name, feature):
        """
        添加人脸特征到数据库
        参数:
            name: 人脸名称/ID
            feature: 人脸特征向量(tensor)
        """
        # 确保face_datas目录存在
        os.makedirs('feature/face_datas', exist_ok=True)  # 创建目录如果不存在
        
        # 将中文名转换为拼音作为文件名
        name_pinyin = ''.join(lazy_pinyin(name)) if any('\u4e00' <= c <= '\u9fff' for c in name) else name
        feature_path = f'feature/face_datas/{name_pinyin}.pt'  # 特征文件路径
        torch.save(feature, feature_path)  # 保存特征到PyTorch文件
        
        # 创建新行并添加到DataFrame
        new_row = pd.DataFrame([[name, feature_path]], 
                            columns=['name', 'feature_path'])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        self.df.to_csv(self.csv_path, index=False)  # 保存更新后的DataFrame到CSV

    def search_face(self, query_feature, threshold=0.68):
        """
        在数据库中搜索匹配的人脸
        参数:
            query_feature: 查询的特征向量
            threshold: 相似度阈值，调整为0.6
        返回:
            匹配的人脸名称，如果没有匹配则返回"Unknown"
        """
        max_sim = 0  # 最大相似度
        best_match = None  # 最佳匹配
        candidates = []  # 候选匹配列表
        
        # 第一轮筛选: 找出所有高于阈值的候选
        for _, row in self.df.iterrows():
            # 加载存储的特征向量
            if isinstance(row['feature_path'], str):
                saved_feature = torch.load(row['feature_path'])  # 从文件加载
            else:
                saved_feature = row['feature_path']  # 直接获取特征
                
            # 确保特征向量是2D张量 [1, 256]
            q_feat = query_feature.unsqueeze(0) if query_feature.dim() == 1 else query_feature
            s_feat = saved_feature.unsqueeze(0) if saved_feature.dim() == 1 else saved_feature
            
            # 计算余弦相似度
            sim = torch.cosine_similarity(
                q_feat,
                s_feat,
                dim=1
            ).item()
            
            # 如果相似度高于阈值，加入候选列表
            if sim > threshold:
                candidates.append((row['name'], sim))
        
        # 如果有多个候选，使用多特征融合策略
        if len(candidates) > 1:
            # 取相似度最高的前3个候选
            candidates.sort(key=lambda x: x[1], reverse=True)
            top3 = candidates[:3]
            
            # 计算平均相似度
            avg_sim = sum(sim for _, sim in top3) / len(top3)
            if avg_sim > 0.7:  # 更高的确认阈值
                best_match = top3[0][0]
                max_sim = avg_sim
        elif candidates:  # 如果只有一个候选
            best_match = candidates[0][0]
            max_sim = candidates[0][1]
        
        self.last_similarity = max_sim  # 记录最后相似度
        return best_match if best_match and max_sim >= threshold else "Unknown"

# 创建全局实例
face_db = FaceDatabase()

# 兼容旧接口的函数
def save_face_to_csv(name, feature):
    face_db.add_face(name, feature)

def check_face(feature):
    return face_db.search_face(feature), face_db.last_similarity
