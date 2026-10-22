from tqdm import tqdm
from utils import *
from models.generator import RTCGenerator
from image_transformer import rotation
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Subset
import argparse

Image.MAX_IMAGE_PIXELS = None
import torch.optim as optim

def main():
    parser = argparse.ArgumentParser(description='RTCG')
    parser.add_argument('--train_dir', default='D:/ImageNet/ImageNet', help='imagenet')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--eps', type=int, default=16)
    parser.add_argument('--model_type', type=str, default='res50', help='Source model = incv3, res50')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--label_flag', type=str, default='N8', help='Label nums: N8, C50')
    parser.add_argument('--nz', type=int, default=16, help='nz')
    parser.add_argument('--save_dir', type=str, default='checkpoints/')
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument("--train_subset_size", type=int, default=325000)
    parser.add_argument("--seed", type=int, default=1111)
    args = parser.parse_args()
    print(args)

    eps = args.eps / 255.
    use_gpu = torch.cuda.is_available()

    if use_gpu:
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda:0")
    print(device)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.model_type == 'incv3':
        scale_size, img_size = 300, 299
        netG = RTCGenerator(cond_channels=4,cond_dim=128,dim=64,n_latent=8, n_heads_latent=8, inception=True)
    else:
        scale_size, img_size = 256, 224
        netG = RTCGenerator(cond_channels=4,cond_dim=128,dim=64, n_latent=8, n_heads_latent=8, inception=False )

    if args.start_epoch > 0:
        netG.load_state_dict(torch.load(args.load_path, map_location=device, weights_only=True))
        print("pretrained checkpoint loaded")
    netG = netG.to(device)

    optimG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))

    train_set = get_data(args.train_dir, scale_size, img_size)

    if args.train_subset_size and args.train_subset_size > 0 and args.train_subset_size < len(train_set):
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(len(train_set), args.train_subset_size, replace=False).tolist()
        train_set = Subset(train_set, indices)
    train_size = len(train_set)
    print('Training data size:', train_size)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=6, pin_memory = True)

    if args.model_type == 'incv3':
        model = torchvision.models.inception_v3( weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1).to(device)
    elif args.model_type == 'res50':
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1).to(device)
    model.eval()

    # class
    label_set = get_classes(args.label_flag)

    criterion = nn.CrossEntropyLoss()

    latent_cond_dict = torch.load('latents.pth', map_location=device , weights_only= False)

    save_dir = os.path.join(args.save_dir, args.model_type)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    log_file = os.path.join(save_dir, "loss_log.txt")

    for epoch in range(args.start_epoch, args.epochs):
        running_loss = 0
        with open(log_file, "a") as log:
            for i, (imgs, gg) in enumerate(tqdm(train_loader)):
                img = imgs[0].to(device)
                img_rot = rotation(img)[0]
                img_aug = imgs[1].to(device)

                np.random.shuffle(label_set)
                label = np.random.choice(label_set, img.size(0))
                cond = torch.stack([latent_cond_dict[j] for j in label], dim=0)
                label = torch.from_numpy(label).long().to(device)


                netG.train()
                optimG.zero_grad()

                noise = netG(x=img, cond=cond, eps=eps)
                noise_rot = netG(x=img_rot, cond=cond, eps=eps)
                noise_aug = netG(x=img_aug, cond=cond, eps=eps)

                adv = noise + img
                adv = torch.clamp(adv, 0.0, 1.0)

                adv_rot = noise_rot + img_rot
                adv_rot = torch.clamp(adv_rot, 0.0, 1.0)

                adv_aug = noise_aug + img_aug
                adv_aug = torch.clamp(adv_aug, 0.0, 1.0)

                adv_out = model(normalize(adv))
                adv_rot_out = model(normalize(adv_rot))
                adv_aug_out = model(normalize(adv_aug))

                loss = criterion(adv_out, label) + criterion(adv_rot_out, label) + criterion(adv_aug_out, label)
                loss.backward()
                optimG.step()

                running_loss += loss.item()

                if i % 100 == 99:
                    print('Epoch: {} \t Batch: {}/{} \t loss: {:.5f}'.format(epoch, i, len(train_loader),running_loss / 100))
                    log.write(f'Epoch: {epoch} Batch: {i}/{len(train_loader)} Loss: {running_loss / 100:.5f}\n')
                    log.flush()
                    running_loss = 0
            if epoch >= args.start_epoch:
                torch.save(netG.state_dict(), '{}/model-{}.pth'.format(save_dir, epoch))


if __name__ == "__main__":
    main()
