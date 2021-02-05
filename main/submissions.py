import os
import time
from os.path import basename

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import dataloader
from preprocess import preproc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# data and model paths
test_data = './diabetic-retinopathy-detection/test/test'
model_path = './weights/full_inceptionv3_AutoWtdCE_2020-12-17_14-16_epoch47.pth'


# dataloader
test_loader = dataloader(None, test_data, preproc(), 'test')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def summision_generate(model, batch_size):

    result = {}
    since = time.time()

    print('-' * 10)

    model.eval()   # Set model to evaluate mode
    batch_iterator = iter(DataLoader(
        test_loader, batch_size, shuffle=False, num_workers=8))

    # for images, labels in test_loader:
    print(len(test_loader))
    
    iteration = int(len(test_loader)/batch_size)
#     with open(f'./submissions/{basename(model_path)[:-4]}.csv', 'w') as f:
#         f.write(f'image,level\n')
    for step in tqdm(range(iteration), desc="Running..."):
        images, labels = next(batch_iterator)

        images = images.to(device)

        # run predictions
        with torch.set_grad_enabled(False):
            outputs = model(images)
            preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
        
#         import pdb; pdb.set_trace()
        if 'image' not in result.keys():
            result['image'] = labels
            result['level'] = preds.tolist()
        else:
            result['image'] += labels
            result['level'] += preds.tolist()
#         res = preds.tolist()
#         with open(f'./submissions/{basename(model_path)[:-4]}.csv', 'a') as f:
#             for b in range(74):
#                 f.write(f'{test_loader.names[b]},{res[b]}\n')
            
    df = pd.DataFrame.from_dict(result) 

    # saving the dataframe
    df.to_csv(f'./submissions/{basename(model_path)[:-4]}.csv', index=False)

    time_elapsed = time.time() - since
    print('Runnning complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


if __name__ == "__main__":
    model = torch.load(model_path)
    model = model.to(device)

    summision_generate(model, batch_size=181)
