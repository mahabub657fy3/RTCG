import os
import argparse
from torch.utils.data import DataLoader
from utils import *
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor
import io
import numpy as np
import torch
import json
from datetime import datetime
import logging
from pathlib import Path

class JPEGFilter:
    def __init__(self, quality=70):
        self.quality = quality

    def __call__(self, images):
        device = images.device
        outputs = []

        for img in images.cpu():
            pil_img = to_pil_image(img)

            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG", quality=self.quality)
            buffer.seek(0)

            compressed = Image.open(buffer).convert("RGB")
            outputs.append(to_tensor(compressed))

        return torch.stack(outputs).to(device)

# Set up logging
def setup_logging():
    log_dir = Path("mydefense/jpeg/jepglog")
    log_dir.mkdir(parents=True, exist_ok=True)  # Fixed: Use Path object and parents=True
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"attack_eval_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

log_file = setup_logging()
logger = logging.getLogger(__name__)

def save_results(args, results):
    result_dir = Path("mydefense/jpeg/results")
    result_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = result_dir / f"results_{timestamp}.json"
    
    data = {
        "timestamp": timestamp,
        "args": vars(args),
        "results": results
    }
    
    with open(result_file, 'w') as f:
        json.dump(data, f, indent=4)
    
    logger.info(f"Results saved to {result_file}")

def main():
    parser = argparse.ArgumentParser(description='TGAF')
    parser.add_argument('--test_dir', default='\gan_n8\model-9', help='Testing Data')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch Size')
    parser.add_argument('--model_t', type=str, default='normal', help='Model under attack: vgg16, vgg19, ..., dense121')
    parser.add_argument('--label_flag', type=str, default='N8', help='Label nums: N8, C20, C50, ...')
    parser.add_argument('--jpeg_quality', type=int, default=70, help='JPEG compression quality (Q), default=70')
    args = parser.parse_args()
    
    logger.info(f"Starting evaluation with arguments: {args}")
    logger.info(f"Logging to file: {log_file}")

    n_class = 1000
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    dic = dict()
    try:
        # Initialize JPEG defense
        jpeg_defense = JPEGFilter(quality=args.jpeg_quality)
        logger.info(f"Initialized JPEG defense with quality {args.jpeg_quality}")
    except Exception as e:
        logger.error(f"Failed to initialize JPEG defense: {str(e)}")
        raise

    if args.model_t == 'all':
        model_name_list = ['vgg16', 'googlenet', 'incv3', 'res152', 'dense121', 'incv4', 'inc_res_v2', 
                          'adv_incv3', 'ens_inc_res_v2', 'res50_sin', 'res50_sin_in', 'res50_sin_fine_in']
    elif args.model_t == 'robust':
        model_name_list = ['adv_incv3', 'ens_inc_res_v2', 'res50_sin', 'res50_sin_in', 
                          'res50_sin_fine_in', 'res50_augmix']
    elif args.model_t == 'normal':
        model_name_list = ["incv3", "vgg16", "googlenet", "res50", "res152", "dense121"]
    else:
        model_name_list = [args.model_t]

    for model_name in model_name_list:
        try:
            logger.info(f"Loading model: {model_name}")
            model_t = load_model(model_name)
            model_t = model_t.to(device)
            model_t.eval()
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            continue

        img_size = 299 if model_name in ['incv3', 'incv4', 'inc_res_v2', 'adv_incv3', 'ens_inc_res_v2'] else 224

        data_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])

        try:
            class_ids = get_classes(args.label_flag)
            logger.info(f"Evaluating on classes: {class_ids}")
        except Exception as e:
            logger.error(f"Failed to get class IDs: {str(e)}")
            continue

        sr = np.zeros(len(class_ids))
        for idx in range(len(class_ids)):
            test_dir = f"{args.test_dir}_t{class_ids[idx]}"
            logger.info(f"Processing test directory: {test_dir}")

            target_acc = 0.
            target_test_size = 0.
            
            try:
                test_set = datasets.ImageFolder(test_dir, data_transform)
                test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, 
                                        num_workers=4, pin_memory=True)
                
                for i, (img, _) in enumerate(test_loader):
                    img = img.to(device)
                    
                    try:
                        img_compressed = jpeg_defense(img.cpu()).to(device)
                        adv_out = model_t(normalize(img_compressed.clone().detach()))
                        target_acc += torch.sum(adv_out.argmax(dim=-1) == (class_ids[idx])).item()
                        target_test_size += img.size(0)
                    except Exception as e:
                        logger.warning(f"Error processing batch {i}: {str(e)}")
                        continue
                
                sr[idx] = target_acc / target_test_size
                logger.info(f"Class {class_ids[idx]} success rate: {sr[idx]:.2%}")
            
            except Exception as e:
                logger.error(f"Error processing test directory {test_dir}: {str(e)}")
                continue
        
        logger.info(f"Model {model_name} - Avg target acc: {sr.mean():.2%} - Test size: {target_test_size}")
        dic[model_name] = sr.mean() * 100

    logger.info("Evaluation completed")
    logger.info(f"Final results: {dic}")
    save_results(args, dic)

if __name__ == "__main__":
    main()