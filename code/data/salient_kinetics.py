from torchvision.transforms.functional import to_tensor
from PIL import Image
from torch import Tensor
import torch
from .kinetics import Kinetics400
from pathlib import Path
from typing import Tuple, List
from saliency.flow.optflow import flow_read
from typing import Tuple, List, Optional
from generate_saliency import method_index

import numpy as np

class SalientKinetics400(Kinetics400):
    """
    Args:
        root (string): Root directory of the Kinetics-400 Dataset.
        prior_root (string, optional): Root directory of the Prior Dataset, 
            if None generate a simple saliency map
        frames_per_clip (int): number of frames in a clip
        step_between_clips (int): number of frames between each clip
        transform (callable, optional): A function/transform that  takes in a TxHxWxC video
            and returns a transformed version.

    Returns:
        video (Tensor[T, H, W, C]): the `T` video frames
        audio (Tensor[K, L]): the audio frames, where `K` is the number of channels
            and `L` is the number of points
        saliency (Tensor[T, H, W, C]): Saliency information of video
        label (int): class of the video clip
    """

    def __init__(self, root, prior_roots: List[Optional[str]], frames_per_clip, step_between_clips=1, frame_rate=None,
                 extensions=('mp4',), transform=None, salient_transform=None, 
                 cached=None, _precomputed_metadata=None):
        super(SalientKinetics400, self).__init__(root, frames_per_clip, 
                                                step_between_clips=step_between_clips,
                                                frame_rate=frame_rate, extensions=extensions, 
                                                transform=transform, cached=cached, 
                                                _precomputed_metadata=_precomputed_metadata)

        self.salient_transform = salient_transform
        self.prior_roots = []
        
        for prior in prior_roots:
            if prior is None:
                self.prior_roots.append(None)
            else:
                path = Path(prior)
                self.prior_roots.append(path)
                if not path.is_dir():
                    raise FileNotFoundError(f'Could not find saliency data at {path}')

    def clip_idx_to_frame(self, clip_location: Tuple[int, int]) -> List:
        video_idx, clip_idx = clip_location

        video_pts = self.video_clips.metadata['video_pts'][video_idx]
        clip_pts = self.video_clips.clips[video_idx][clip_idx]
        
        # Map video_pts values to indices, theses indices are the frame ids
        to_frame = { pts.item(): i for i, pts in enumerate(video_pts) }
        frames = [to_frame[pts.item()] for pts in clip_pts]
        return frames

    def load_optical_flow_frame(self, path: Path) -> Tensor:
        flow = flow_read(str(path))
        return to_tensor(flow)

    def load_frame(self, path: Path) -> Tensor:
        channels = self.get_number_of_channels(path)
        with open(str(path), 'rb') as f:
            img = Image.open(f)
            if channels == 1:
                img = img.convert('L')
            elif channels == 2:
                img = img.convert('RGB')

        img = to_tensor(img)

        if channels == 2:
            # Discard B channel, img shape: (h, w, 2)
            img = img[:2, :, :].permute(1, 2, 0)

        return img.squeeze()

    def generate_saliency_clip(self, shape: torch.Size) -> Tensor:
        """ Used to generate simple saliency maps """
        T, W, H, C = shape
        frame = torch.cartesian_prod(torch.arange(W), torch.arange(H))
        frame = frame.reshape(W, H, 2).float()
        center = torch.tensor([W/2, H/2]).unsqueeze(0)
        distance = torch.cdist(frame, center).squeeze()

        distance -= distance.min()
        distance /= distance.max()
        distance = 1 - distance
        return torch.stack(T * [distance])

    def scale_saliency(self, shape: torch.Size, saliency: torch.Tensor):
        # Scale saliency to size of video if they are not the same
        if saliency.shape[-2:] != shape[1:3]:
            # HACK: I accidentally transposed one dataset partially which 
            # does not need interpolation but it does need transposing
            vid_size = torch.tensor(shape[1:3]) 
            sal_size = torch.tensor(saliency.shape[-2:])

            if (vid_size == torch.flip(sal_size, [0])).all():
                saliency = saliency.permute(0, 2, 1)
            else:
                correct_size = (shape[1], shape[2])
                
                # Interpolate needs the channel dimension
                if len(saliency.shape) <= 3:
                    saliency = saliency.unsqueeze(1)
                saliency = torch.nn.functional.interpolate(saliency, size=correct_size)
                saliency = saliency.squeeze()
        return saliency

    def get_frame_offset(self, prior):
        for key, (_, one_indexed) in method_index.items():
            if key in str(prior):
                return int(one_indexed)
        raise Exception('Could not infer frame offset')

    def get_number_of_channels(self, prior):
        if 'flow' in str(prior):
            return 2
        return 1

    def get_saliency_clip(self, clip_location: Tuple[int, int], shape) -> Tensor:
        """
        Get (precomputed) saliency clip
        """
        video_idx, clip_idx = clip_location

        video_path = self.video_clips.metadata['video_paths'][video_idx]
        video_path = Path(video_path)
        video_name = video_path.stem

        # Maintain folder structure of original dataset
        subfolders = video_path.relative_to(Path(self.root).parent).parent
        
        frames = self.clip_idx_to_frame(clip_location)
        priors = []
        for prior in self.prior_roots:
            if prior is not None:
                saliencies = []
                for frame in frames:
                    cached_folder = prior / subfolders / video_name
                    offset = self.get_frame_offset(prior)
                    cached_file = cached_folder / f'{frame + offset}.jpg'
                    
                    if self.get_number_of_channels(prior) == 2:
                        cached_file = cached_file.with_suffix('.flo')

                    if cached_file.is_file():
                        if self.get_number_of_channels(prior) == 2:
                            saliency_frame = self.load_optical_flow_frame(cached_file)
                        else:
                            saliency_frame = self.load_frame(cached_file)
                    else:
                        raise FileNotFoundError(f'Could not load frame {cached_file} from video {video_idx}.')
                    saliencies.append(saliency_frame)
                saliency = torch.stack(saliencies)
                saliency = self.scale_saliency(shape, saliency)
            else:
                saliency = self.generate_saliency_clip(shape)
            priors.append(saliency)
        
        if len(priors) > 1:
            # TODO: Mean or Max?
            return torch.mean(torch.stack(priors), dim=0)

        return priors[0]


    def __getitem__(self, idx):
        success = False
        while not success:
            try:
                video, audio, info, video_idx = self.video_clips.get_clip(idx)

                # This information is needed for saliency caching
                clip_location = self.video_clips.get_clip_location(idx)
                saliency = self.get_saliency_clip(clip_location, video.shape)
                
                success = True
            except Exception as e:
                print('skipped idx', idx, e)
                idx = np.random.randint(self.__len__())
        
        # saliency = self.get_saliency_clip(video, clip_location)
        label = self.samples[video_idx][1]

        # The random state is kept constant for the two transforms, this
        # makes sure RandomResizedCrop is applied the same way in both 
        # video and saliency maps.
        random_state = torch.get_rng_state()

        if self.transform is not None:
            video = self.transform(video)

        if self.salient_transform is not None:
            torch.set_rng_state(random_state)
            saliency = self.salient_transform(saliency)

        return video, audio, saliency, label
