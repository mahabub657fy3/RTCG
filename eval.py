import argparse
from torch.utils.data import DataLoader
from utils import *
from models.generator import RTCGenerator

def main():
    parser = argparse.ArgumentParser(description='RTCG')
    parser.add_argument('--data_dir', default='/neurips2017_dev', help='ImageNet validation data')
    parser.add_argument('--is_nips', action='store_true', default=True,  help='Evaluation on NIPS data')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--eps', type=int, default=16, help='Perturbation budget')
    parser.add_argument('--model_type', type=str, default='incv3',  help='Source model = incv3, res50')
    parser.add_argument('--load_path', type=str, default='checkpoints/', help='Load path')
    parser.add_argument('--label_flag', type=str, default='N8', help='Label nums: N8')
    parser.add_argument('--nz', type=int, default=16, help='nz')
    parser.add_argument('--save_dir', type=str, default='results/', help='Dir to save adv images')
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    eps = args.eps/255.
    n_class = 1000

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    if args.model_type == 'incv3':
        scale_size = 300
        img_size = 299
    else:
        scale_size = 256
        img_size = 224

    if args.model_type == 'incv3':
        netG = RTCGenerator(cond_channels=4, cond_dim=128, dim=64, n_latent=8, n_heads_latent=8,)
    else:
        netG = RTCGenerator( cond_channels=4, cond_dim=128, dim=64, n_latent=8, n_heads_latent=8,)

    netG.load_state_dict(torch.load(args.load_path, map_location=device))
    netG = netG.to(device)
    netG.eval()

    data_transform = transforms.Compose([
        transforms.Resize(scale_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])

    test_set = datasets.ImageFolder(args.data_dir, data_transform)
    test_size = len(test_set)
    print('Test data size:', test_size)

    if args.is_nips:
        print('is_nips')
        test_set = fix_labels_nips(args, test_set, pytorch=True)
    else:
        test_set = fix_labels(args, test_set)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    class_ids = get_classes(args.label_flag)

    print(class_ids)

    latent_cond_dict = torch.load('latents.pth', map_location=device, weights_only=False)

    for idx in range(len(class_ids)):
        for i, (img, label) in enumerate(test_loader):
            img = img.to(device)
            cond = latent_cond_dict[class_ids[idx]].unsqueeze(0).expand(img.size()[0], -1, -1, -1)
            noise = netG(img, cond, eps=eps)
            adv = noise + img

            adv = torch.min(torch.max(adv, img - eps), img + eps)
            adv = torch.clamp(adv, 0.0, 1.0)

            save_imgs = adv.detach().cpu()
            for j in range(len(save_imgs)):
                g_img = transforms.ToPILImage('RGB')(save_imgs[j])
                output_dir = '{}/gan_n8/{}_t{}/images'.format(args.save_dir, args.load_path.split('/')[-1].split('.pth')[0], class_ids[idx])
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                g_img.save(os.path.join(output_dir, '{}_{}.png'.format(class_ids[idx], i * args.batch_size + j)))
if __name__ == "__main__":
    main()
