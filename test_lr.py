from dataloader import BatchedImages, LabeledImage, NYUv2Dataset
from model import DepthAndUncertaintyModel
from opt import config_parser
import torch
import random, os
from torch.utils.data import Dataset, DataLoader
import yaml
from logger import get_git_revision_short_hash, get_git_status, get_git_diff, init_log, write_to_log
from load_model import build_model
import torchvision 
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

def loss_func(x, y):
    return torch.mean(torch.square(x-y))

def uncertainty_loss(x, y):
    prediction = x[:,:,:,0:1]
    variance = x[:,:,:,1:2]
    y = y.permute(0,2,3,1)
    mse = torch.square(y - prediction)
    return torch.mean(0.5 * torch.exp(-variance) * mse + 0.5 * variance), torch.mean(mse)

    
def train_epoch(args, model, optimizer, dataloader, i, device, scheduler):
    losses = []
    mses = []

    for i, batch in enumerate(dataloader):
        batch = BatchedImages(batch.rgb.to(device), batch.label.to(device))
        loss, mse = uncertainty_loss(model(batch), batch.label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
  
        losses.append(loss)
        mses.append(mse)
        scheduler.step()

    write_to_log(f"MEAN TRAIN LOSS: {torch.mean(torch.stack(losses))}")
    write_to_log(f"MEAN TRAIN MSE: {torch.mean(torch.stack(mses))}")
   
def eval_epoch(args, model, dataloader, i, device, logfolder):
    losses = []
    mses = []
    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            batch = BatchedImages(batch.rgb.to(device), batch.label.to(device))
            preds = model(batch)
            loss, mse = uncertainty_loss(preds, batch.label)
            losses.append(loss)
            mses.append(mse)
    
            if (i * args.bs) % 5 == 0:
                pred = preds[0].permute(2, 0, 1).cpu()
                var = torch.exp(pred[1:2, :, :])
                var = (var * 50).to(torch.uint8) 
                mu = pred[0:1, :, :]
                mu = torch.clamp((mu * 5).to(torch.uint8), 0, 255) 

                rgb = batch.rgb[0].cpu()
                rgb = (rgb * 255).to(torch.uint8)

                gt = batch.label[0].cpu()
                gt = (gt * 5).to(torch.uint8)
                torchvision.io.write_png(var, os.path.join(logfolder, f"var_{i}.png"))
                torchvision.io.write_png(mu, os.path.join(logfolder, f"mean_{i}.png"))
                torchvision.io.write_png(rgb, os.path.join(logfolder, f"rgb_{i}.png"))
                torchvision.io.write_png(gt, os.path.join(logfolder, f"gt{i}.png"))

    write_to_log(f"VAL LOSS: {torch.mean(torch.stack(losses))}")
    write_to_log(f"VAL MSE: {torch.mean(torch.stack(mses))}")

def collate_fn(labeled_imgs: list[LabeledImage]):
    batched_imgs = BatchedImages(
    rgb=torch.stack([labeled_img.rgb for labeled_img in labeled_imgs]),
    label=torch.stack([labeled_img.label for labeled_img in labeled_imgs]))
    return batched_imgs

def create_dataloader(args):
    train_dataset = NYUv2Dataset(mat_file_path='nyu_depth_v2_labeled.mat', splits_path='nyuv2_splits.mat', mode='train')
    test_dataset = NYUv2Dataset(mat_file_path='nyu_depth_v2_labeled.mat', splits_path='nyuv2_splits.mat', mode='test')
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)
    val_dataloader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)
    return train_dataloader, val_dataloader

def run_training(args):
    model = torch.nn.Linear(1, 1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    it_per_epoch = 100

    scheduler1 = LinearLR(optimizer, start_factor=0.1, total_iters=args.epochs * it_per_epoch // 10)
    scheduler2 = CosineAnnealingLR(optimizer, args.epochs * it_per_epoch // 10 * 9, 0.01)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[args.epochs * it_per_epoch // 10])

    for i in range(args.epochs + 1):
        for j in range(it_per_epoch):
            scheduler.step()
        print(f"Epoch {i} {optimizer.param_groups[0]['lr']}")
    
     
if __name__ == '__main__':
    args = config_parser()
    run_training(args)
