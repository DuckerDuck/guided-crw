import argparse
import numpy as np
from utils.test_utils import vis_pose
from pathlib import Path
from data.jhmdb import load_image
import cv2

from PIL import Image

def read_vallist(args):

    with open(args.vallist) as f:
        lines = f.readlines()

    result = []
    for line in lines:
        line = line.strip()
        imgs, vids = line.split(' ')
        result.append((imgs, vids))
    return result


def resize_img(img, args):
    ht, wd = img.shape[1], img.shape[2]
    if args.imgsize > 0:
        ratio  = 1.0 
        if ht <= wd:
            img = cv2.resize( img, (int(args.imgsize * ratio), args.imgsize))
        else:
            img = cv2.resize( img, (args.imgsize, int(args.imgsize * ratio)))
    return img


def main(args):
    map_scale = np.array([8, 8])

    vids = read_vallist(args)
    output_dir = Path(args.output)
    if not output_dir.exists():
        output_dir.mkdir()

    for i, (imgs, lbls) in enumerate(vids):
        print(i, imgs)
        joint_file = Path(args.joints) / (str(i) + '.dat')
        video_dir = output_dir / str(i)

        if not video_dir.exists():
            video_dir.mkdir()

        predres  = np.load(joint_file, allow_pickle=True)
        loc, joints, frames = predres.shape

        img_base = Path(imgs)
        if not img_base.exists():
            print(f'Background folder {img_base} not found!')
            return

        for f in range(frames):
            img_path = img_base / (str(f+1).zfill(5) + '.png')
            if not img_path.exists():
                print(f'Could not find image {img_path}!')
                return
            
            coords = predres[:, :, f].squeeze()
            img = load_image(str(img_path)).permute(1,2,0).numpy() * 255
            img = resize_img(img, args)
            pose_map = vis_pose(img, coords * map_scale[..., None])

            pose_map = pose_map.astype(np.uint8)
            output_file = video_dir / (str(f+1).zfill(5) + '.jpg')
            with open(output_file, 'w') as file_ptr:
                img = Image.fromarray(pose_map)
                img.save(file_ptr, format='jpeg', quality=90)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Skeletons of JHMDB results')
    parser.add_argument('joints', help='Path to .dat file containing joint coordinates')
    parser.add_argument('vallist', help='JHMDB vallist')
    parser.add_argument('--output', default='./JHMDB_visualized/', help='Path to plot folder')
    parser.add_argument('--imgsize', default=320, help='Same as cropSize in test.py')
    
    args = parser.parse_args()
    print(args)
    main(args)
