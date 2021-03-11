import argparse
import torch
from data.salient_kinetics import SalientKinetics400
from train import _get_cache_path
from pathlib import Path

def generate(args):
    cached = None
    if args.cache_dataset:
        cache_path = _get_cache_path(args.cache_dataset_path, args)
        print(cache_path)
        if Path(cache_path).is_file():
            print('Using cached dataset')
            dataset, _ = torch.load(cache_path)
            cached = dict(video_paths=dataset.video_clips.video_paths,
                    video_fps=dataset.video_clips.video_fps,
                    video_pts=dataset.video_clips.video_pts)

    dataset = SalientKinetics400(
                    args.data_path,
                    args.saliency_path,
                    frames_per_clip=1,
                    step_between_clips=1,
                    transform=None,
                    salient_transform=None,
                    extensions=('mp4'),
                    frame_rate=None,
                    _precomputed_metadata=cached
                )
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=None, num_workers=args.workers//2,
        pin_memory=True, collate_fn=None)

    print(f'Total number of videos: {len(data_loader)}')

    prev_progress = -1
    data_generator = enumerate(data_loader)
    while True:
        i = 0
        try:
            i, _ = next(data_generator)
        except Exception as e:
            print('skipped video clip', i, str(e))

        progress = int((i / len(data_loader)) * 100)
        if prev_progress != progress:
            print(f'{progress}% videos processed')
            prev_progress = progress
        




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Saliency Map Cache Generator')

    parser.add_argument('--data-path', default='/../kinetics/', help='Path to dataset')
    parser.add_argument('--saliency-path', default='./saliency_cache/',
        help='Path to saliency cache')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument( "--cache-dataset", dest="cache_dataset", help="Cache the datasets for quicker initialization. It also serializes the transforms", action="store_true")
    parser.add_argument( "--cache-dataset-path", dest="cache_dataset_path", help="Path of cached dataset", default=None)
 
    args = parser.parse_args()

    generate(args)