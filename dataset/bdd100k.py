import os

import cv2
import numpy as np
from PIL import Image
import sys
sys.path.append('/kaggle/working/adversarial-patch-transferability')

import torch
from dataset.base_dataset import BaseDataset

class BDD100K(BaseDataset):
    """
    BDD100K Dataset for semantic segmentation.
    
    BDD100K contains 100k diverse driving videos with rich annotations.
    The semantic segmentation subset has 10k images with instance segmentation labels.
    Labels are compatible with Cityscapes (19 classes).
    
    Dataset structure:
    - root/
        - images/
            - train/
            - val/
            - test/
        - drivable_maps/
            - train/
            - val/
        - semantic_masks/
            - train/
            - val/
    
    or
    
    - root/
        - bdd100k_images/
        - bdd100k_labels_release/
    """
    
    def __init__(self, 
                 root, 
                 list_path,
                 num_classes=19,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=255, 
                 base_size=2048, 
                 crop_size=(512, 1024),
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225],
                 bd_dilate_size=4):

        super(BDD100K, self).__init__(ignore_label, base_size,
                crop_size, scale_factor, mean, std,)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip
        
        # Load file list from text file or auto-discover
        list_file_path = os.path.join(root, list_path) if not root.endswith('/') else root + list_path
        if os.path.exists(list_file_path):
            self.img_list = [line.strip().split() for line in open(list_file_path)]
        else:
            print(f"Warning: List file '{list_file_path}' not found. Auto-discovering files...")
            self.img_list = self._auto_discover_files()

        self.files = self.read_files()

        # BDD100K label mapping (same as Cityscapes for compatibility)
        # Raw class IDs (0-33) mapped to evaluation classes (0-18)
        self.label_mapping = {-1: ignore_label, 0: ignore_label, 
                              1: ignore_label, 2: ignore_label, 
                              3: ignore_label, 4: ignore_label, 
                              5: ignore_label, 6: ignore_label, 
                              7: 0, 8: 1, 9: ignore_label, 
                              10: ignore_label, 11: 2, 12: 3, 
                              13: 4, 14: ignore_label, 15: ignore_label, 
                              16: ignore_label, 17: 5, 18: ignore_label, 
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                              25: 12, 26: 13, 27: 14, 28: 15, 
                              29: ignore_label, 30: ignore_label, 
                              31: 16, 32: 17, 33: 18}
        
        # Class weights (estimated from BDD100K distribution)
        # BDD100K has different class distributions than Cityscapes
        # These are approximate weights for the 19 classes
        self.class_weights = torch.FloatTensor([0.82, 0.95, 0.88, 1.05, 
                                        1.02, 0.98, 0.97, 1.08,
                                        0.85, 1.03, 0.96, 0.99, 
                                        1.12, 0.92, 1.09, 1.11, 
                                        1.08, 1.15, 1.05])
        
        self.bd_dilate_size = bd_dilate_size
        
        # BDD100K specific class names (19 classes)
        self.class_names = [
            'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
            'traffic light', 'traffic sign', 'vegetation', 'terrain',
            'sky', 'person', 'rider', 'car', 'truck', 'bus', 
            'train', 'motorcycle', 'bicycle'
        ]
    
    def _auto_discover_files(self):
        """
        Auto-discover image and label files when list file is not available.
        Scans root/images/{split}/ and root/labels/{split}/ directories.
        Handles multiple BDD100K label naming patterns.
        """
        # Extract split name from list_path (e.g., 'train.txt' -> 'train')
        split = os.path.splitext(os.path.basename(self.list_path))[0]
        
        img_dir = os.path.join(self.root, 'images', split)
        label_dir = os.path.join(self.root, 'labels', split)
        
        # Try alternative structures
        if not os.path.exists(img_dir):
            # Try: root/bdd100k_images/10k/{split}/
            img_dir = os.path.join(self.root, 'bdd100k_images', '10k', split)
            label_dir = os.path.join(self.root, 'bdd100k_labels_release', 'sem_seg', 'masks', split)
        
        if not os.path.exists(img_dir):
            # Try: root/images/{split}/
            img_dir = os.path.join(self.root, 'images', split)
            # Try different label directories
            for label_subdir in ['labels', 'sem_seg', 'semantic_masks', 'drivable_maps']:
                test_label_dir = os.path.join(self.root, label_subdir, split)
                if os.path.exists(test_label_dir):
                    label_dir = test_label_dir
                    break
        
        if not os.path.exists(img_dir):
            raise ValueError(f"Image directory not found: {img_dir}\n"
                           f"Please check dataset structure or provide a list file.")
        
        print(f"Scanning directory: {img_dir}")
        
        # Get all image files (BDD100K uses .jpg typically)
        images = []
        for root_dir, dirs, files in os.walk(img_dir):
            for f in files:
                if f.endswith('.jpg') or f.endswith('.png'):
                    rel_path = os.path.relpath(os.path.join(root_dir, f), self.root)
                    images.append(rel_path)
        
        images = sorted(images)
        
        # Common BDD100K label naming patterns
        label_patterns = [
            (lambda x: x.replace('.jpg', '_drivable_id.png').replace('.png', '_drivable_id.png'), 'drivable_id'),
            (lambda x: x.replace('.jpg', '_train_id.png').replace('.png', '_train_id.png'), 'train_id'),
            (lambda x: x.replace('.jpg', '.png'), 'same_name'),
            (lambda x: x.replace('.jpg', '_label.png').replace('.png', '_label.png'), 'label_suffix'),
        ]
        
        img_list = []
        for img_path in images:
            # Try to find corresponding label
            label_found = False
            
            for pattern_func, pattern_name in label_patterns:
                label_path = pattern_func(img_path)
                label_path = label_path.replace('images', 'labels')
                label_path = label_path.replace('bdd100k_images', 'bdd100k_labels_release/sem_seg/masks')
                
                if os.path.exists(os.path.join(self.root, label_path)):
                    img_list.append([img_path, label_path])
                    label_found = True
                    break
            
            # If no label found, check if it's test set
            if not label_found:
                if 'test' in split:
                    img_list.append([img_path])
                    label_found = True
            
            if not label_found:
                print(f"Warning: Label not found for {img_path}, skipping...")
        
        print(f"Auto-discovered {len(img_list)} image-label pairs in '{split}' split")
        
        if len(img_list) == 0:
            raise ValueError(f"No valid image-label pairs found in {img_dir}")
        
        return img_list
    
    def read_files(self):
        """
        Read image and label file paths from list file.
        List file format:
            image_path label_path  (for train/val)
            image_path             (for test)
        """
        files = []
        if 'test' in self.list_path:
            for item in self.img_list:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                files.append({
                    "img": image_path[0],
                    "name": name,
                })
        else:
            for item in self.img_list:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name
                })
        return files
        
    def convert_label(self, label, inverse=False):
        """
        Convert between raw class IDs and evaluation class IDs.
        
        Args:
            label: label map
            inverse: if True, convert from evaluation IDs back to raw IDs
        """
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        
        # Load image (BGR format from OpenCV)
        image = cv2.imread(os.path.join(self.root, item["img"]),
                           cv2.IMREAD_COLOR)
        size = image.shape

        # Test time: return only image
        if 'test' in self.list_path:
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))
            return image.copy(), np.array(size), name

        # Training/validation time: load label and perform augmentation
        label = cv2.imread(os.path.join(self.root, item["label"]),
                           cv2.IMREAD_GRAYSCALE)
        label = self.convert_label(label)

        # Generate sample with augmentation (multi-scale, flip, edge)
        image, label, edge = self.gen_sample(image, label, 
                                self.multi_scale, self.flip, edge_size=self.bd_dilate_size)

        return image.copy(), label.copy(), edge.copy(), np.array(size), name

    
    def single_scale_inference(self, config, model, image):
        """
        Run single-scale inference on image.
        """
        pred = self.inference(config, model, image)
        return pred


    def save_pred(self, preds, sv_path, name):
        """
        Save predictions to disk.
        
        Args:
            preds: prediction tensor (B, C, H, W)
            sv_path: save directory
            name: image names
        """
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))
