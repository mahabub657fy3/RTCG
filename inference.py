import os
import argparse
from torch.utils.data import DataLoader
from utils import *

def main():
    parser = argparse.ArgumentParser(description='TGAF')
    parser.add_argument('--test_dir', default='results/gan_n8/model-9', help='Testing Data')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch Size')
    parser.add_argument('--model_t',type=str, default= 'normal',  help ='Model under attack : vgg16, vgg19, ..., dense121')
    parser.add_argument('--label_flag', type=str, default='N8', help='Label nums: N8, C20, C50, ...')
    args = parser.parse_args()
    print(args)

    n_class = 1000
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    dic = dict()

    if args.model_t == 'all':
        model_name_list = ['vgg16', 'googlenet', 'incv3', 'res152', 'dense121', 'incv4', 'inc_res_v2', 'adv_incv3', 'ens_inc_res_v2', 'res50_sin', 'res50_sin_in', 'res50_sin_fine_in']

    elif args.model_t == 'robust':
        model_name_list = ['adv_incv3', 'ens_inc_res_v2', 'res50_sin', 'res50_sin_in', 'res50_sin_fine_in','res50_augmix']
    elif args.model_t == 'normal':
        model_name_list = ["incv3", "vgg16", "googlenet", "res50", "res152", "dense121"]
    elif args.model_t == 'vit':
        model_name_list = ['vit_base_patch16_224', 'pit_b_224','visformer_small', 'swin_tiny_patch4_window7_224']
    else:
        model_name_list = [args.model_t]

    for model_name in model_name_list:
        model_t = load_model(model_name)

        model_t = model_t.to(device)
        model_t.eval()

        if model_name in ['incv3', 'incv4', 'inc_res_v2', 'adv_incv3', 'ens_inc_res_v2']:
            img_size = 299
        else:
            img_size = 224

        data_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])


        class_ids = get_classes(args.label_flag)

        sr = np.zeros(len(class_ids))
        for idx in range(len(class_ids)):
            test_dir = '{}_t{}'.format(args.test_dir, class_ids[idx])

            target_acc = 0.
            target_test_size = 0.
            test_set = datasets.ImageFolder(test_dir, data_transform)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                                    pin_memory=True)
            for i, (img, _) in enumerate(test_loader):
                img = img.to(device)
                adv_out = model_t(normalize(img.clone().detach()))
                target_acc += torch.sum(adv_out.argmax(dim=-1) == (class_ids[idx])).item()
                target_test_size += img.size(0)
            sr[idx] = target_acc / target_test_size
            print('sr: {}'.format(sr))
        print('model:{} \t target acc:{:.2%}\t target_test_size:{}'.format(model_name, sr.mean(), target_test_size))
        dic[model_name] = sr.mean() * 100
    print(dic)
if __name__ == "__main__":
    main()