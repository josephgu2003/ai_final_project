from dataloader import BatchedImages, LabeledImage, NYUv2Dataset
from model import DepthAndUncertaintyModel
from opt import config_parser
import torch
import random, os
from torch.utils.data import Dataset, DataLoader
import yaml
from logger import get_git_revision_short_hash, get_git_status, get_git_diff, init_log, write_to_log
from load_model import build_model
import torchvision.transforms as T
from visualizer import generate_visuals
 
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

def train_epoch(args, model, optimizer, dataloader, i, device, loss_func):
    losses = []
    mses = []

    for i, batch in enumerate(dataloader):
        batch = BatchedImages(batch.rgb.to(device), batch.label.to(device))
        preds = model(batch)[0]
        loss = loss_func(preds, batch.label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss)
        
        with torch.no_grad():
            mses.append(mse(preds, batch.label))
    
    write_to_log(f"TRAIN LOSS: {torch.mean(torch.stack(losses))}")
    write_to_log(f"TRAIN MSE: {torch.mean(torch.stack(mses))}")
    
def eval_epoch(args, model, dataloader, epoch, device, logfolder, dropout_samples, visualize, loss_func):
    losses = []
    mses = []
    for idx, batch in enumerate(dataloader):
        with torch.no_grad():
            batch = BatchedImages(batch.rgb.to(device), batch.label.to(device))
            preds, preds_var = model(batch, dropout_samples)
                                
            loss = loss_func(preds, batch.label)
            losses.append(loss)
            mses.append(mse(preds, batch.label))
            
            if visualize:
                generate_visuals(batch, preds, preds_var, epoch, idx, logfolder, args=args,)
            
    write_to_log(f"VAL LOSS: {torch.mean(torch.stack(losses))}")
    write_to_log(f"VAL MSE: {torch.mean(torch.stack(mses))}")

def collate_fn(labeled_imgs: list[LabeledImage]):
    batched_imgs = BatchedImages(
    rgb=torch.stack([labeled_img.rgb for labeled_img in labeled_imgs]),
    label=torch.stack([labeled_img.label for labeled_img in labeled_imgs]))
    return batched_imgs

def uncertainty_loss(x, y):
    prediction = x[:,:,:,0:1]
    variance = x[:,:,:,1:2]
    y = y.permute(0,2,3,1)
    mse = torch.square(y - prediction)
    return torch.mean(0.5 * torch.exp(-variance) * mse + 0.5 * variance)

def mse(x, y):
    prediction = x[:,:,:,0:1]
    y = y.permute(0,2,3,1)
    return torch.mean(torch.square(prediction - y))

def create_dataloader(args):
    train_dataset = NYUv2Dataset(mat_file_path='nyu_depth_v2_labeled.mat', splits_path='nyuv2_splits.mat', mode='train')
    test_dataset = NYUv2Dataset(mat_file_path='nyu_depth_v2_labeled.mat', splits_path='nyuv2_splits.mat', mode='test')
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)
    val_dataloader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)
    return train_dataloader, val_dataloader

def run_training(args):
    set_random_seed(args.seed)
    

    logfolder = os.path.join(args.base_dir, args.exp_name)

    # init log file
    os.makedirs(logfolder, exist_ok=True)
   
    init_log(os.path.join(logfolder, 'log.txt'))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    write_to_log(str(device))
    args_dict = vars(args)

    with open(os.path.join(logfolder, 'saved_config.yaml'), 'w') as f:
        yaml.dump(args_dict, f)
    
    with open(os.path.join(logfolder, 'git_info.txt'), 'w') as f:
        f.write(get_git_status())
        f.write('\n')
        f.write('\n')
        f.write('\n')
        f.write(get_git_diff())
        f.write('\n')
        f.write('\n')
        f.write('\n')
        f.write(get_git_revision_short_hash())

    model, errors = build_model(args, device)

    write_to_log(str(errors))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_dataloader, val_dataloader = create_dataloader(args)
    
    for i in range(1, args.epochs + 1):
        write_to_log(f"TRAIN EPOCH {i}:")
        train_epoch(args, model, optimizer, train_dataloader, i, device, loss_func=uncertainty_loss if args.use_aleatoric else mse)
        write_to_log(f"VAL EPOCH {i}:")
        
        if (i == 1) or (i % args.val_every == 0):
            eval_epoch(args, model, val_dataloader, i, device, logfolder, args.dropout_samples, i in args.vis_interval,
                       loss_func=uncertainty_loss if args.use_aleatoric else mse)

        write_to_log("Saving model to output dir!")   
        ckpt = {'args': args, 'state_dict': model.state_dict(), 'epoch': i}
        torch.save(ckpt, os.path.join(logfolder, 'last_model.pth')) 
    
    write_to_log("Done!")
     
if __name__ == '__main__':
    args = config_parser()
    run_training(args)
