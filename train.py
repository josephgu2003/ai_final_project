from dataloader import BatchedImages, LabeledImage
from opt import config_parser
import torch
import random, os
from torch.utils.data import Dataset, DataLoader

class CustomImageDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 4096

    def __getitem__(self, idx):
        rgb = torch.zeros(3, 32, 32)
        label = torch.zeros(1, 32, 32)

        img = LabeledImage(
            rgb,
            label,
            'fake_path')

        return img
    

class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ff = torch.nn.Linear(3, 1)

    def forward(self, x: BatchedImages):
        bchw = x.rgb
        bhwc = bchw.permute(0, 2, 3, 1)
        bhwd = self.ff(bhwc)
        return bhwd.permute(0, 3, 1, 2)
    
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

def loss_func(x, y):
    return torch.mean(torch.square(x-y))
    
def train_epoch(args, model, optimizer, dataloader, i):
    for i, batch in enumerate(dataloader):
        loss = loss_func(model(batch), batch.label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss)
        
        # log the loss
    
def eval_epoch(args, model, dataloader, i):
    losses = []
    for i, batch in enumerate(dataloader):
        loss = loss_func(model(batch), batch.label)
        losses.append(loss)

    print("VAL LOSS: ", torch.mean(torch.stack(losses)))

def collate_fn(labeled_imgs: list[LabeledImage]):
    batched_imgs = BatchedImages(
    rgb=torch.stack([labeled_img.rgb for labeled_img in labeled_imgs]),
    label=torch.stack([labeled_img.label for labeled_img in labeled_imgs]))
    return batched_imgs

def create_dataloader(args):
    dataset = CustomImageDataset()
    train_dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=False, collate_fn=collate_fn)
    return train_dataloader, val_dataloader

def run_training(args):
    print('Hello world!')
    print(args)
    set_random_seed(args.seed)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    model = CustomModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_dataloader, val_dataloader = create_dataloader(args)
    
    for i in range(args.epochs):
        print(f"TRAIN EPOCH {i}:")
        train_epoch(args, model, optimizer, train_dataloader, i)
        print(f"VAL EPOCH {i}:")
        eval_epoch(args, model, val_dataloader, i)
        
    print("Done!")
     
if __name__ == '__main__':
    args = config_parser()
    run_training(args)