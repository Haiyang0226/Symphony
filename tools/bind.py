import torch
from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

class VideoQAProcessor:
    def __init__(self, device: str = None, cache_dir: str = './cache_dir'):
        """
        初始化视频问答处理类
        
        参数:
        - device: 计算设备 ('cuda:X' 或 'cpu')，如果为None则随机选择0-3号GPU
        - cache_dir: 模型缓存目录
        """
        if device is None:
            # 随机选择0-3号GPU
            gpu_id = random.randint(0, 3)
            self.device = torch.device(f'cuda:{gpu_id}')
        else:
            self.device = torch.device(device)
        self.cache_dir = cache_dir
        
        # 模型配置
        self.clip_type = {
            # 'video': 'LanguageBind_Video',
            #'audio': 'LanguageBind_Audio',
            'image': 'LanguageBind_Image'
        }
        self.pretrained_ckpt = '/home/web_server/antispam/project/zhouhongyun/models/LanguageBind/LanguageBind_Image'
        
        # 组件初始化
        self.model = None
        self.tokenizer = None
        self.transforms = None
        
        # 日志配置
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def initialize_models(self):
        """初始化模型、tokenizer和转换器"""
        
        # 加载模型
        self.model = LanguageBind(
            clip_type=self.clip_type, 
            cache_dir=self.cache_dir
        ).to(self.device).eval()
        
        # 加载tokenizer
        tokenizer_cache = Path(self.cache_dir) / 'tokenizer_cache_dir'
        tokenizer_cache.mkdir(parents=True, exist_ok=True)
        

        self.tokenizer = LanguageBindImageTokenizer.from_pretrained(
            self.pretrained_ckpt, 
            cache_dir=str(tokenizer_cache)
        )
        # 创建模态转换器
        self.transforms = {
            mod: transform_dict[mod](self.model.modality_config[mod])
            for mod in self.clip_type.keys()
        }
        
        self.logger.info("模型初始化完成")

    def extract_image_features(self, image_paths: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        批量提取图像特征
        
        参数:
        - image_paths: 图像路径列表
        - batch_size: 处理批大小            
        返回:
        - 图像特征张量
        """
        if not self.model:
            self.initialize_models()
            
        if not image_paths:
            self.logger.warning("没有可处理的图像")
            return torch.Tensor()
            
        self.logger.info(f"开始提取图像特征: {len(image_paths)}张图像")
        
        # 分批次处理图像
        all_image_feats = []
        total_batches = (len(image_paths) + batch_size - 1) // batch_size
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            self.logger.debug(f"处理批次 {i//batch_size+1}/{total_batches} ({len(batch_paths)}张图像)")
            
            # 处理图像
            image_inputs = to_device(
                self.transforms['image'](batch_paths), 
                self.device
            )
            
            with torch.no_grad():
                embeddings = self.model({'image': image_inputs})
                all_image_feats.append(embeddings['image'].cpu())
        
        # 合并所有图像特征
        image_features = torch.cat(all_image_feats, dim=0)
        
        self.logger.info("图像特征提取完成")
        return image_features


    def extract_features(
        self, 
        frames: List[str], 
        questions: List[str],
        batch_size: int = 100,
        modality: str = 'image',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批量提取图像和文本特征
        
        参数:
        - frames: 帧路径列表
        - questions: 问题文本列表
        - batch_size: 处理批大小
        
        返回:
        - 图像特征张量
        - 文本特征张量
        """
        query = questions
        if not self.model or not self.tokenizer:
            self.logger.error("模型未初始化")
            raise RuntimeError("模型未初始化")
            
        if not frames:
            self.logger.warning("没有可处理的帧")
            return torch.Tensor(), torch.Tensor()
            
        #self.logger.info(f"开始特征提取: {len(frames)}帧 | {len(questions)}个问题")
        
        # 预处理文本
        if isinstance(query[0], str) and os.path.exists(query[0]):
            # 图像查询
            self.logger.info("使用图像查询")
            query_image_features = self.extract_image_features(query)
            text_inputs = None

        elif isinstance(query, (str, list)):
            tokenized_text = self.tokenizer(
                questions, 
                max_length=77, 
                padding='max_length',
                truncation=True, 
                return_tensors='pt'
            )
            text_inputs = to_device(tokenized_text, self.device)

        
        # 分批次处理帧
        all_frame_feats = []
        total_batches = (len(frames) + batch_size - 1) // batch_size
        
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            #self.logger.debug(f"处理批次 {i//batch_size+1}/{total_batches} ({len(batch)}帧/视频clip)")
            
            frame_inputs = to_device(
                self.transforms[modality](batch), 
                self.device
            )
            
            with torch.no_grad():
                if text_inputs != None:
                    inputs = {modality: frame_inputs, 'language': text_inputs}
                else:
                    inputs = {modality: frame_inputs}
                embeddings = self.model(inputs)
                all_frame_feats.append(embeddings[modality].cpu())
        
        
        # 合并所有图像特征
        frame_features = torch.cat(all_frame_feats, dim=0)
        if text_inputs != None:
            text_features = embeddings['language'].cpu()
            return frame_features, text_features
        else:
            return frame_features, query_image_features

    def calculate_similarity_topk(
        self, 
        frame_features: torch.Tensor, 
        text_features: torch.Tensor,
        top_k: int = 100,
        modality: str = 'image',
    ) -> Dict[int, List[Dict]]:
        """
        计算相似度并获取TopK结果
        
        参数:
        - frame_features: 帧特征张量
        - text_features: 文本特征张量
        - top_k: 返回的top结果数量
        
        返回:
        - 按问题索引组织的topK结果
        """
        if frame_features.nelement() == 0 or text_features.nelement() == 0:
            return {}
            
        # 计算相似度矩阵 (frames x questions)
        similarity = frame_features @ text_features.T
 
        # 获取每个问题对应的TopK帧
        topk_values, topk_indices = torch.topk(similarity, k=top_k, dim=0, largest=True)
        
        # 按问题组织结果
        results = {}
        for q_idx in range(text_features.shape[0]):
            q_results = []
            for rank in range(top_k):
                frame_idx = topk_indices[rank, q_idx].item()
                q_results.append({
                    'rank': rank + 1,
                    'frame_idx': frame_idx,
                    'similarity': topk_values[rank, q_idx].item()
                })
            results[q_idx] = q_results
            
        #self.logger.info(f"计算完成: {len(results)}个问题的top{top_k}相似度")
        return results

    def process_video_qa(
        self,
        frames_list: List[str],
        questions_list: str,
        modality: str = 'image',
        batch_size: int = 20,
        top_k: int = 10
    ) -> Dict[int, List[Dict]]:

        # modality = "video"
        
        # 步骤1: 加载数据
        self.frames_list = frames_list
        if len(frames_list)<top_k:
            return frames_list
        
        if not frames_list or not questions_list:
            self.logger.error("数据加载失败，无法继续")
            return {}
            
        # 步骤2: 初始化模型（如果未初始化）
        if not self.model:
            self.initialize_models()
            
        # 步骤3: 处理指定问题

        # 步骤4: 提取特征
        frame_feats, text_feats = self.extract_features(
            frames_list, questions_list, batch_size, modality
        )
        
        # 步骤5: 计算相似度
        results = self.calculate_similarity_topk(
            frame_feats, text_feats, top_k, modality
        )

        top_k_frames = []

        
        for q_idx, q_results in results.items():
            #print(f"\n问题 {q_idx} 的前 {len(q_results)} 相关帧/视频片段:")
            for result in q_results:
                frame_idx = result['frame_idx']
                # 确保索引不越界
                if 0 <= frame_idx < len(frames_list):
                    filename = os.path.basename(frames_list[frame_idx])
                    top_k_frames.append(frames_list[frame_idx])

        #print('top_k_frames:', top_k_frames)
        return top_k_frames