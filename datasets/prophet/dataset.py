import torch
import numpy as np
import os
from datasets.prophet.prepare import *

class ProphetDataset(torch.utils.data.Dataset):
    def __init__(self, split, encoding ,device, block_size):
        super().__init__()
        prepare(encoding)
        # memmap : accessing small segments of large files on disk, 
        #          without reading the entire file into memory.
        # Might slow me down, but I'm poor
        if(split == 'train'):
            self.data = np.memmap( os.path.join(dir_path, f'bin/train-{encoding}.bin'), 
                                  dtype=np.uint16, mode='r')
        elif(split == 'val'):
            self.data = np.memmap( os.path.join(dir_path, f'bin/val-{encoding}.bin'), 
                                  dtype=np.uint16, mode='r')
        else:
            raise ValueError(f"Only train and val splits are supported, got {split}")
        self.block_size = block_size
        self.device = device

    def __len__(self):
        return len(self.data) - self.block_size - 1
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            try:
                x = torch.stack(
                    [torch.from_numpy((self.data[i:i+self.block_size]).astype(np.int64)) 
                        for i in range(*idx.indices(len(self.data))) ])
                # use only the token after the block as rest are already available through x
                y = torch.stack(
                    [torch.from_numpy( np.array( (self.data[i+1+self.block_size]).astype(np.int64) ) ) 
                        for i in range(*idx.indices(len(self.data))) ]) 
            except:
                import time
                print(f"Invalid slice, {idx} {len(self.data)} {self.block_size}")
                time.sleep(1)
        else:
            x = torch.from_numpy((self.data[idx:idx+self.block_size]).astype(np.int64))
            y = torch.from_numpy( np.array( (self.data[idx+1+self.block_size]).astype(np.int64) ) )
        if self.device == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y