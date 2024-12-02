from dataloader import BatchedImages
import torch
import os
from logger import write_to_log
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import sobel

def compute_depth_edges(depth):
    depth_x = sobel(depth, axis=0)  # Horizontal gradient
    depth_y = sobel(depth, axis=1)  # Vertical gradient
    edges = np.hypot(depth_x, depth_y)  # Gradient magnitude
    edges = (edges / edges.max() * 255).astype(np.uint8)  # Normalize
    return edges

def generate_visuals(batch, preds, preds_var, epoch, idx, logfolder, args):
    batch_size = batch.rgb.size(0)
    for j in range(1): # okay so visualizing the entire test set takes too much disk memory
        rgb = batch.rgb[j].permute(1, 2, 0).cpu().numpy()  # [C, H, W] -> [H, W, C]
        gt = batch.label[j, 0].cpu().numpy()  # [1, H, W] -> [H, W]
        pred = preds[j, :, :, 0].cpu().numpy()  # Prediction mean
        dropout_var = preds_var[j, :, :, 0:1].cpu().numpy()

        pred_norm = (pred - pred.min()) / (pred.max() - pred.min())
        dropout_var_norm = (dropout_var - dropout_var.min()) / (dropout_var.max()- dropout_var.min())

        # Compute edges from predicted depth
        pred_edges = compute_depth_edges(pred)
        pred_edges_norm = (pred_edges / np.max(pred_edges)) * 255  # Normalize

        # Depth and uncertainty combined visualization
        cmap = plt.get_cmap('plasma')
        depth_color = cmap(pred_norm)[:, :, :3]  # Apply colormap, discard alpha
       # combined_visual = depth_color * (1 - var_norm[:, :, None])  # Adjust brightness

        # Normalize ground truth for separate visualization
        gt_norm = (gt / np.max(gt)) * 255

        fig, axs = plt.subplots(1, 5, figsize=(30, 5))
        axs = axs.ravel()

        axs[0].imshow(rgb)
        axs[0].set_title('RGB Image')
        axs[0].axis('off')

        axs[1].imshow(gt_norm, cmap='plasma')
        axs[1].set_title('Ground Truth Depth')
        axs[1].axis('off')

        axs[2].imshow(pred_norm, cmap='plasma')
        axs[2].set_title('Predicted Depth')
        axs[2].axis('off')

        if args.use_aleatoric:
            var = torch.exp(preds[j, :, :, 1]).cpu().numpy()  # Uncertainty/variance
            var_norm = (var - var.min()) / (var.max() - var.min())
            axs[3].imshow(var_norm, cmap='hot')
            axs[3].set_title('Aleatoric Uncertainty')
            axs[3].axis('off')
        
        if args.use_epistemic:
            axs[4].imshow(dropout_var_norm, cmap='hot')
            axs[4].set_title('Epistemic Uncertainty')
            axs[4].axis('off')
        
        save_folder = os.path.join(logfolder, f"visuals_epoch_{epoch}")
        
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
            
        save_path = os.path.join(save_folder, f"batch_{idx * batch_size + j}.png")
        plt.savefig(save_path)
        plt.close(fig)
