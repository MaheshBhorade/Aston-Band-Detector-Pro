import torch
import clip
import faiss
import numpy as np
from PIL import Image

class CLIPEngine:
    def __init__(self, model_name="ViT-B/32", device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading CLIP model {model_name} on {self.device}...")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        
        # Faiss Index
        self.index = None
        self.id_to_name = {}

    def get_features(self, pil_image):
        image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().astype('float32')

    def add_to_index(self, prototype_dict):
        """
        Adds new prototypes to the existing Faiss index.
        """
        if not prototype_dict:
            return
            
        # Get dimension from first entry
        first_feat = next(iter(prototype_dict.values()))
        d = first_feat.shape[1]
        
        # Initialize index if it doesn't exist
        if self.index is None:
            print("Initializing new Faiss IndexFlatIP...")
            self.index = faiss.IndexFlatIP(d)
        
        start_idx = self.index.ntotal if self.index is not None else 0
        features_list = []
        for i, (name, feat) in enumerate(prototype_dict.items()):
            features_list.append(feat)
            self.id_to_name[start_idx + i] = name
            
        features_matrix = np.vstack(features_list)
        # Normalize for cosine similarity
        faiss.normalize_L2(features_matrix)
        self.index.add(features_matrix)
        print(f"Added {len(prototype_dict)} prototypes. Total now: {len(self.id_to_name)}")

    def search(self, current_features, top_k=1):
        if self.index is None or self.index.ntotal == 0:
            return 0, None
            
        # Normalize search vector
        faiss.normalize_L2(current_features)
        scores, indices = self.index.search(current_features, top_k)
        
        best_idx = indices[0][0]
        best_score = scores[0][0]
        
        if best_idx == -1:
            return 0, None
            
        return best_score, self.id_to_name.get(best_idx, "Unknown")
