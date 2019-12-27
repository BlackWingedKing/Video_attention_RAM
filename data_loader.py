import torch #TODO: only load in needed parts of the torch library for the module
from torchvision.datasets import UCF101 #Use pre-made torchvision dataloaders for UCF101

def ucf101_dataset(root='../data/UCFvideo', annotation_path='../data/UCFlabels', frames_per_clip=16, step_between_clips=8, **kwargs):

    ucf_dataset = UCF101(root, annotation_path, frames_per_clip, step_between_clips, **kwargs)
    return ucf_dataset
    
def ucf101_dataloader(dataset, batch_size=4, shuffle=True, num_workers=1):

    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

test_set = ucf101_dataloader(ucf101_dataset(train=False))

test_set = iter(test_set)
test_set.next()