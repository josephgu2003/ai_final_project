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

def uncertainty_loss(x, y):
    prediction = x[:,:,:,0:1]
    variance = x[:,:,:,1:2]
    y = y.permute(0,2,3,1)
    mse = torch.square(y - prediction)
    return torch.mean(0.5 * torch.exp(-variance) * mse + 0.5 * variance), torch.mean(mse)

def generate_visuals(args, model, dataloader, i, device, logfolder):
    losses = []
    mses = []
    for idx, batch in enumerate(dataloader):
        with torch.no_grad():
            batch = BatchedImages(batch.rgb.to(device), batch.label.to(device))
            preds = model(batch)
            loss, mse = uncertainty_loss(preds, batch.label)
            losses.append(loss)
            mses.append(mse)
            
            if idx % 50 == 0: # Change later to generate more images
                rgb = batch.rgb[0].permute(1, 2, 0).cpu().numpy()  # [C, H, W] -> [H, W, C]
                gt = batch.label[0, 0].cpu().numpy()  # [1, H, W] -> [H, W]
                pred = preds[0, :, :, 0].cpu().numpy()  # Prediction mean
                var = torch.exp(preds[0, :, :, 1]).cpu().numpy()  # Uncertainty/variance

                pred_norm = (pred - pred.min()) / (pred.max() - pred.min())
                var_norm = (var - var.min()) / (var.max() - var.min())

                # Compute edges from predicted depth
                pred_edges = compute_depth_edges(pred)
                pred_edges_norm = (pred_edges / np.max(pred_edges)) * 255  # Normalize

                # Depth and uncertainty combined visualization
                cmap = plt.get_cmap('plasma')
                depth_color = cmap(pred_norm)[:, :, :3]  # Apply colormap, discard alpha
                combined_visual = depth_color * (1 - var_norm[:, :, None])  # Adjust brightness

                # Normalize ground truth for separate visualization
                gt_norm = (gt / np.max(gt)) * 255

                fig, axs = plt.subplots(2, 3, figsize=(30, 10))
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

                axs[3].imshow(var_norm, cmap='hot')
                axs[3].set_title('Uncertainty')
                axs[3].axis('off')

                axs[4].imshow(combined_visual)
                axs[4].set_title('Depth + Uncertainty')
                axs[4].axis('off')

                axs[5].imshow(pred_edges_norm, cmap='gray')
                axs[5].set_title('Depth Edges')
                axs[5].axis('off')

                save_path = os.path.join(logfolder, f"edge_visuals_new/visuals_epoch_{i}_batch_{idx}.png")
                plt.savefig(save_path)
                plt.close(fig)

    write_to_log(f"VAL LOSS: {torch.mean(torch.stack(losses))}")
    write_to_log(f"VAL MSE: {torch.mean(torch.stack(mses))}")