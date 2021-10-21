from torch.utils.data import Dataset
import random
import torch
import glob
import os

from utils import load_image, uniform_temporal_subsample, load_dicom

class DataRetriever(Dataset):
    def __init__(self, patient_path, paths, targets, n_frames, img_size, transform=None):
        self.patient_path = patient_path
        self.paths = paths
        self.targets = targets
        self.n_frames = n_frames
        self.img_size = img_size
        self.transform = transform
          
    def __len__(self):
        return len(self.paths)
    
    def read_video(self, vid_paths):
        video = [load_image(path, (self.img_size, self.img_size)) for path in vid_paths]
        if self.transform:
            seed = random.randint(0,99999)
            for i in range(len(video)):
                random.seed(seed)
                video[i] = self.transform(image=video[i])["image"]
        
        video = [torch.tensor(frame, dtype=torch.float32) for frame in video]
        if len(video)==0:
            video = torch.zeros(self.n_frames, self.img_size, self.img_size)
        else:
            video = torch.stack(video) # T * C * H * W
        return video
    
    def __getitem__(self, index):
        _id = self.paths[index]
        patient_path = os.path.join(self.patient_path, f'{str(_id).zfill(5)}/')

        channels = []
        for t in ["FLAIR", "T1w", "T1wCE", "T2w"]:
            t_paths = sorted(
                glob.glob(os.path.join(patient_path, t, "*")), 
                key=lambda x: int(x[:-4].split("-")[-1]),
            )
            num_samples = self.n_frames
            if len(t_paths) < num_samples:
                in_frames_path = t_paths
            else:
                in_frames_path = uniform_temporal_subsample(t_paths, num_samples)
            
            channel = self.read_video(in_frames_path)
            if channel.shape[0] == 0:
                channel = torch.zeros(num_samples, self.img_size, self.img_size)
            channels.append(channel)
            
        channels = torch.stack(channels).transpose(0,1)
        y = torch.tensor(self.targets[index], dtype=torch.float)
        return {"X": channels.float(), "y": y}

class TestDataRetriever(Dataset):
    def __init__(self, patient_path, paths, n_frames, img_size, transform=None):
        self.patient_path = patient_path
        self.paths = paths
        self.n_frames = n_frames
        self.img_size = img_size
        self.transform = transform
          
    def __len__(self):
        return len(self.paths)
    
    def read_video(self, vid_paths):
        video = [load_dicom(path, self.img_size) for path in vid_paths]
        if len(video)==0:
            video = torch.zeros(self.n_frames, self.img_size, self.img_size)
        else:
            video = torch.stack(video) # T * C * H * W
        return video
    
    def __getitem__(self, index):
        _id = self.paths[index]
        patient_path = os.path.join(self.patient_path, f'{str(_id).zfill(5)}/')
        channels = []
        for t in ["FLAIR","T1w", "T1wCE", "T2w"]:
            t_paths = sorted(
                glob.glob(os.path.join(patient_path, t, "*")), 
                key=lambda x: int(x[:-4].split("-")[-1]),
            )
            num_samples = self.n_frames
            if len(t_paths) < num_samples:
                in_frames_path = t_paths
            else:
                in_frames_path = uniform_temporal_subsample(t_paths, num_samples)
            
            channel = self.read_video(in_frames_path)
            if channel.shape[0] == 0:
                print("1 channel empty")
                channel = torch.zeros(num_samples, self.img_size, self.img_size)
            channels.append(channel)
        
        channels = torch.stack(channels).transpose(0,1)
        return {"X": channels.float(), "id": _id}