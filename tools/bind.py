import torch
from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
import random
import config

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

class VideoQAProcessor:
    def __init__(self, device: str = None, cache_dir: str = './cache_dir'):
        """
        Initialize the Video QA processor class
        
        Parameters:
        - device: Computing device ('cuda:X' or 'cpu'), if None, randomly select GPU 0-3
        - cache_dir: Model cache directory
        """
        if device is None:
            # Randomly select GPU 0-3
            gpu_id = random.randint(0, 3)
            self.device = torch.device(f'cuda:{gpu_id}')
        else:
            self.device = torch.device(device)
        self.cache_dir = cache_dir
        
        # Model configuration
        self.clip_type = {
            # 'video': 'LanguageBind_Video',
            #'audio': 'LanguageBind_Audio',
            'image': 'LanguageBind_Image'
        }
        self.pretrained_ckpt = config.LANGUAGEBIND_MODEL_PATH
        
        # Component initialization
        self.model = None
        self.tokenizer = None
        self.transforms = None
        
        # Logging configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def initialize_models(self):
        """Initialize model, tokenizer and transforms"""
        
        # Load model
        self.model = LanguageBind(
            clip_type=self.clip_type, 
            cache_dir=self.cache_dir
        ).to(self.device).eval()
        
        # Load tokenizer
        tokenizer_cache = Path(self.cache_dir) / 'tokenizer_cache_dir'
        tokenizer_cache.mkdir(parents=True, exist_ok=True)
        

        self.tokenizer = LanguageBindImageTokenizer.from_pretrained(
            self.pretrained_ckpt, 
            cache_dir=str(tokenizer_cache)
        )
        # Create modality transforms
        self.transforms = {
            mod: transform_dict[mod](self.model.modality_config[mod])
            for mod in self.clip_type.keys()
        }
        
        self.logger.info("Model initialization completed")

    def extract_image_features(self, image_paths: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        Extract image features in batches
        
        Parameters:
        - image_paths: List of image paths
        - batch_size: Processing batch size            
        Returns:
        - Image feature tensor
        """
        if not self.model:
            self.initialize_models()
            
        if not image_paths:
            self.logger.warning("No images to process")
            return torch.Tensor()
            
        self.logger.info(f"Starting image feature extraction: {len(image_paths)} images")
        
        # Process images in batches
        all_image_feats = []
        total_batches = (len(image_paths) + batch_size - 1) // batch_size
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            self.logger.debug(f"Processing batch {i//batch_size+1}/{total_batches} ({len(batch_paths)} images)")
            
            # Process images
            image_inputs = to_device(
                self.transforms['image'](batch_paths), 
                self.device
            )
            
            with torch.no_grad():
                embeddings = self.model({'image': image_inputs})
                all_image_feats.append(embeddings['image'].cpu())
        
        # Merge all image features
        image_features = torch.cat(all_image_feats, dim=0)
        
        self.logger.info("Image feature extraction completed")
        return image_features


    def extract_features(
        self, 
        frames: List[str], 
        questions: List[str],
        batch_size: int = 100,
        modality: str = 'image',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract image and text features in batches
        
        Parameters:
        - frames: List of frame paths
        - questions: List of question texts
        - batch_size: Processing batch size
        
        Returns:
        - Image feature tensor
        - Text feature tensor
        """
        query = questions
        if not self.model or not self.tokenizer:
            self.logger.error("Model not initialized")
            raise RuntimeError("Model not initialized")
            
        if not frames:
            self.logger.warning("No frames to process")
            return torch.Tensor(), torch.Tensor()
            
        #self.logger.info(f"开始特征提取: {len(frames)}帧 | {len(questions)}个问题")
        
        # Preprocess text
        if isinstance(query[0], str) and os.path.exists(query[0]):
            # Image query
            self.logger.info("Using image query")
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

        
        # Process frames in batches
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
        
        
        # Merge all image features
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
        Calculate similarity and get TopK results
        
        Parameters:
        - frame_features: Frame feature tensor
        - text_features: Text feature tensor
        - top_k: Number of top results to return
        
        Returns:
        - TopK results organized by question index
        """
        if frame_features.nelement() == 0 or text_features.nelement() == 0:
            return {}
            
        # Calculate similarity matrix (frames x questions)
        similarity = frame_features @ text_features.T
 
        # Get TopK frames for each question
        topk_values, topk_indices = torch.topk(similarity, k=top_k, dim=0, largest=True)
        
        # Organize results by question
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
        
        # Step 1: Load data
        self.frames_list = frames_list
        if len(frames_list)<top_k:
            return frames_list
        
        if not frames_list or not questions_list:
            self.logger.error("Data loading failed, cannot continue")
            return {}
            
        # Step 2: Initialize model (if not initialized)
        if not self.model:
            self.initialize_models()
            
        # Step 3: Process specified questions

        # Step 4: Extract features
        frame_feats, text_feats = self.extract_features(
            frames_list, questions_list, batch_size, modality
        )
        
        # Step 5: Calculate similarity
        results = self.calculate_similarity_topk(
            frame_feats, text_feats, top_k, modality
        )

        top_k_frames = []

        
        for q_idx, q_results in results.items():
            #print(f"\n问题 {q_idx} 的前 {len(q_results)} 相关帧/视频片段:")
            for result in q_results:
                frame_idx = result['frame_idx']
                # Ensure index is within bounds
                if 0 <= frame_idx < len(frames_list):
                    filename = os.path.basename(frames_list[frame_idx])
                    top_k_frames.append(frames_list[frame_idx])

        #print('top_k_frames:', top_k_frames)
        return top_k_frames