import argparse
import numpy as np
from utils.test_utils import vis_pose
from pathlib import Path
from data.jhmdb import load_image

from PIL import Image

def main(args):
    predres  = np.load(args.joints, allow_pickle=True)
    loc, joints, frames = predres.shape

    map_scale = np.array([8, 8])
    print(map_scale)

    img_base = Path(args.background)
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
        pose_map = vis_pose(img, coords * map_scale[..., None])

        pose_map = pose_map.astype(np.uint8)

        with open(args.output + str(f+1).zfill(5) + '.jpg', 'w') as file_ptr:
            img = Image.fromarray(pose_map)
            img.save(file_ptr, format='jpeg', quality=90)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Skeletons of JHMDB results')
    parser.add_argument('joints', help='Path to .dat file containing joint coordinates')
    parser.add_argument('background', help='Path to jpg file with background')
    parser.add_argument('--output', default='./JHMDB_visualized/', help='Path to plot folder')
    
    args = parser.parse_args()
    print(args)
    main(args)