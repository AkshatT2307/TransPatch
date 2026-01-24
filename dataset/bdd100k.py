import os
import logging

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
                 bd_dilate_size=4,
                 use_color_labels=True):

        super(BDD100K, self).__init__(ignore_label, base_size,
                crop_size, scale_factor, mean, std,)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes
        self.use_color_labels = use_color_labels

        self.multi_scale = multi_scale
        self.flip = flip
        
        # Load file list from text file or auto-discover
        list_file_path = os.path.join(root, list_path) if not root.endswith('/') else root + list_path
        if os.path.exists(list_file_path):
            self.img_list = [line.strip().split() for line in open(list_file_path)]
        else:
            logging.getLogger().debug(f"List file '{list_file_path}' not found. Auto-discovering files...")
            self.img_list = self._auto_discover_files()

        self.files = self.read_files()
        
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
        Handles BDD100K seg structure: root/images/{split}/ and root/labels/{split}/
        """
        # Extract split name from list_path (e.g., 'train.txt' -> 'train')
        split = os.path.splitext(os.path.basename(self.list_path))[0]
        
        # BDD100K seg structure: bdd100k_seg/bdd100k/seg/images/{split}/ and labels/{split}/
        img_dir = os.path.join(self.root, 'images', split)
        
        # Choose label directory based on use_color_labels parameter
        if self.use_color_labels:
            label_dir = os.path.join(self.root, 'color_labels', split)
            alt_label_dir = os.path.join(self.root, 'labels', split)
        else:
            label_dir = os.path.join(self.root, 'labels', split)
            alt_label_dir = os.path.join(self.root, 'color_labels', split)
        
        # Verify directories exist
        if not os.path.exists(img_dir):
            raise ValueError(f"Image directory not found: {img_dir}\n"
                           f"Expected structure: {self.root}/images/{split}/\n"
                           f"Please check dataset path or provide a list file.")
        
        if not os.path.exists(label_dir):
            # Try alternative label directory
            label_dir = alt_label_dir
            if not os.path.exists(label_dir):
                primary_dir = 'color_labels' if self.use_color_labels else 'labels'
                alt_dir = 'labels' if self.use_color_labels else 'color_labels'
                raise ValueError(f"Label directory not found.\n"
                               f"Tried: {self.root}/{primary_dir}/{split}/\n"
                               f"Tried: {self.root}/{alt_dir}/{split}/\n"
                               f"Please check dataset structure.")
        
        logging.getLogger().debug(f"Images directory: {img_dir}")
        logging.getLogger().debug(f"Labels directory: {label_dir}")
        
        # Get all image files from images/{split}/ directory
        images = []
        img_files = sorted(os.listdir(img_dir))
        for f in img_files:
            if f.endswith('.jpg') or f.endswith('.png'):
                images.append(f)
        
        logging.getLogger().debug(f"Found {len(images)} images in {split} split")
        
        if len(images) == 0:
            raise ValueError(f"No image files (.jpg or .png) found in {img_dir}")
        
        # Match images with labels
        # BDD100K labels typically have _train_id.png suffix
        img_list = []
        label_files = set(os.listdir(label_dir))
        
        for img_name in images:
            img_base = os.path.splitext(img_name)[0]
            
            # Try different label naming patterns
            possible_label_names = [
                f"{img_base}_train_id.png",
                f"{img_base}_label.png",
                f"{img_base}.png",
            ]
            
            label_found = False
            for label_name in possible_label_names:
                if label_name in label_files:
                    img_path = os.path.join('images', split, img_name)
                    label_path = os.path.join('labels', split, label_name)
                    # Adjust if using color_labels
                    if 'color_labels' in label_dir:
                        label_path = os.path.join('color_labels', split, label_name)
                    
                    img_list.append([img_path, label_path])
                    label_found = True
                    break
            
            if not label_found:
                if 'test' in split:
                    # Test set may not have labels
                    img_path = os.path.join('images', split, img_name)
                    img_list.append([img_path])
                elif len(img_list) < 3:
                    # Only warn for first few
                    logging.getLogger().debug(f"No label found for {img_name}")
        
        logging.getLogger().debug(f"Matched {len(img_list)} image-label pairs")
        
        if len(img_list) == 0:
            raise ValueError(f"No valid image-label pairs found.\n"
                           f"Images: {img_dir} ({len(images)} files)\n"
                           f"Labels: {label_dir} ({len(label_files)} files)\n"
                           f"Check label naming pattern (expected: imagename_train_id.png)")
        
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
            save_img = Image.fromarray(preds[i])
            save_img.save(os.path.join(sv_path, name[i]+'.png'))
