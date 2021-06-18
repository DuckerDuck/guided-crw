
import torchvision
import skimage

import torch
from torchvision import transforms

import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor

IMG_MEAN = (0.4914, 0.4822, 0.4465)
IMG_STD  = (0.2023, 0.1994, 0.2010)
NORM = [transforms.ToTensor(), 
        transforms.Normalize(IMG_MEAN, IMG_STD)]


class MapTransform(object):
    def __init__(self, transforms, pil_convert=True):
        self.transforms = transforms
        self.pil_convert = pil_convert

    def __call__(self, vid):
        if isinstance(vid, Image.Image):
            return np.stack([self.transforms(vid)])
        
        if isinstance(vid, torch.Tensor) and self.pil_convert:
            vid = vid.numpy()

        if self.pil_convert:
            x = np.stack([np.asarray(self.transforms(Image.fromarray(v))) for v in vid])
            return x
        else:
            return np.stack([self.transforms(v) for v in vid])
    
def n_patches(x, n, transform, shape=(64, 64, 3), scale=[0.2, 0.8]):
    ''' unused '''
    if shape[-1] == 0:
        shape = np.random.uniform(64, 128)
        shape = (shape, shape, 3)

    crop = transforms.Compose([
        lambda x: Image.fromarray(x) if not 'PIL' in str(type(x)) else x,
        transforms.RandomResizedCrop(shape[0], scale=scale)
    ])    

    if torch.is_tensor(x):
        x = x.numpy().transpose(1,2, 0)
    
    P = []
    for _ in range(n):
        xx = transform(crop(x))
        P.append(xx)

    return torch.cat(P, dim=0)


def patch_grid(transform, shape=(64, 64, 3), stride=[0.5, 0.5], convert_to_pil=True):
    stride = np.random.random() * (stride[1] - stride[0]) + stride[0]
    stride = [int(shape[0]*stride), int(shape[1]*stride), shape[2]]

    def to_pil(x):
        if x.shape[2] == 3:
            return Image.fromarray(x)
        y = Image.fromarray(np.squeeze(x, axis=2))
        return y

    pre_transform = to_pil
    post_transform = lambda x: x
    if not convert_to_pil:
        pre_transform = to_tensor
        post_transform = lambda x: x.numpy()
    
    spatial_jitter = transforms.Compose([
        pre_transform,
        transforms.RandomResizedCrop(shape[0], scale=(0.7, 0.9)),
        post_transform
    ])

    def aug(x):
        if torch.is_tensor(x):
            x = x.numpy().transpose(1, 2, 0)
        elif 'PIL' in str(type(x)):
            x = np.array(x)#.transpose(2, 0, 1)
        
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=2)
        winds = skimage.util.view_as_windows(x, shape, step=stride)
        winds = winds.reshape(-1, *winds.shape[-3:])

        jitters = [spatial_jitter(w) for w in winds]
        P = [transform(j) for j in jitters]
        return torch.cat(P, dim=0)

    return aug


def get_frame_aug(frame_aug, patch_size):
    train_transform = []

    if 'cj' in frame_aug:
        _cj = 0.1
        train_transform += [
            #transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(_cj, _cj, _cj, 0),
        ]
    if 'flip' in frame_aug:
        train_transform += [transforms.RandomHorizontalFlip()]

    train_transform += NORM
    train_transform = transforms.Compose(train_transform)

    print('Frame augs:', train_transform, frame_aug)

    if 'grid' in frame_aug:
        aug = patch_grid(train_transform, shape=np.array(patch_size))
    else:
        aug = train_transform

    return aug


def get_frame_transform(frame_transform_str, img_size):
    tt = []
    fts = frame_transform_str
    norm_size = torchvision.transforms.Resize((img_size, img_size))

    if 'crop' in fts:
        tt.append(torchvision.transforms.RandomResizedCrop(
            img_size, scale=(0.8, 0.95), ratio=(0.7, 1.3), interpolation=2),)
    else:
        tt.append(norm_size)

    if 'cj' in fts:
        _cj = 0.1
        # tt += [#transforms.RandomGrayscale(p=0.2),]
        tt += [transforms.ColorJitter(_cj, _cj, _cj, 0),]

    if 'flip' in fts:
        tt.append(torchvision.transforms.RandomHorizontalFlip())

    print('Frame transforms:', tt, fts)

    return tt

def get_salient_frame_aug(frame_aug, patch_size, channels=1):
    train_transform = []

    if 'flip' in frame_aug:
        train_transform += [transforms.RandomHorizontalFlip()]
    
    train_transform += [transforms.ToTensor()]
    if channels == 2:
        train_transform += [lambda x: x.permute(1, 0, 2)]
    train_transform = transforms.Compose(train_transform)

    if 'grid' in frame_aug:
        shape = np.array(patch_size)
        # Saliency maps have varying channel sizes
        shape[2] = channels
        aug = patch_grid(train_transform, shape=shape, convert_to_pil=channels == 1)
    else:
        aug = train_transform

    return aug

def get_salient_frame_transform(fts, img_size):
    tt = []
    norm_size = torchvision.transforms.Resize((img_size, img_size))

    if 'crop' in fts:
        tt.append(torchvision.transforms.RandomResizedCrop(
            img_size, scale=(0.8, 0.95), ratio=(0.7, 1.3), interpolation=2),)
    else:
        tt.append(norm_size)

    if 'flip' in fts:
        tt.append(torchvision.transforms.RandomHorizontalFlip())

    return tt

def get_resized_transform(main_transform, scale):
    """
    Return transform that applies a resize before and after
    the main_transform argument.
    """
    if scale >= 1:
        return None

    def transforms(video):
        original_scale = video.shape[1], video.shape[0]
        scaled_scale = int(original_scale[0] * scale), int(original_scale[1] * scale)
        pre = torchvision.transforms.Compose([
            lambda x: x.permute(2, 0, 1),
            torchvision.transforms.Resize(scaled_scale),
            lambda x: x.permute(2, 1, 0),
        ])

        post = torchvision.transforms.Compose([
            lambda x: x.unsqueeze(-1).permute(2, 0, 1),
            torchvision.transforms.Resize(original_scale),
            lambda x: x.permute(2, 1, 0).squeeze(),
            ])
        # Apply transforms
        stage1 = pre(video)
        stage2 = main_transform(stage1)
        stage3 = post(stage2)
        return stage3
    
    return transforms
    

def get_train_saliency_transform(args):
    """
    Identical to get_train_transforms except:
        - No color normalization
        - No color jitter
        - Single channel
    """
    norm_size = torchvision.transforms.Resize((args.img_size, args.img_size))

    frame_transform = get_salient_frame_transform(args.frame_transforms, args.img_size)
    
    channels = 1
    if len(args.prior_dataset) == 1 and args.prior_dataset[0] == 'flow':
        channels = 2
    
    frame_aug = get_salient_frame_aug(args.frame_aug, args.patch_size, channels=channels)
    frame_aug = [frame_aug] if args.frame_aug != '' else []
    
    transform = frame_transform + frame_aug

    train_transform = MapTransform(
            torchvision.transforms.Compose(transform), pil_convert=channels == 1
        )

    plain = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        norm_size, 
        transforms.ToTensor(),
    ])

    def with_orig(x):
        x = train_transform(x), plain(x[0])
        return x

    return with_orig


def get_train_transforms(args):
    norm_size = torchvision.transforms.Resize((args.img_size, args.img_size))

    frame_transform = get_frame_transform(args.frame_transforms, args.img_size)
    frame_aug = get_frame_aug(args.frame_aug, args.patch_size)
    frame_aug = [frame_aug] if args.frame_aug != '' else NORM
    
    transform = frame_transform + frame_aug

    train_transform = MapTransform(
            torchvision.transforms.Compose(transform)
        )

    plain = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        norm_size, 
        *NORM,
    ])

    def with_orig(x):
        x = train_transform(x), \
            plain(x[0]) if 'numpy' in str(type(x[0])) else plain(x[0].permute(2, 0, 1))

        return x

    return with_orig

