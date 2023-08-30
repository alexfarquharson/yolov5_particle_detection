from utils.general import xyxy2xywh
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def filter_preds(preds, stats, confidence_threshold, iou_threshold):
    # Takes in preds and stats and filters to remove if conf low or iou (from stats) low
    filtered_predictions = []
    iou_threshold_map = {0.5 : 0, 0.55 : 1, 0.6 : 2, 0.65 : 3, 0.7 : 4, 0.75 : 5, 0.8 : 6, 0.85 : 7, 0.9 : 8, 0.95 : 9}
    iou_threshold = iou_threshold_map[iou_threshold]
    
    # filter to remove predictions below iou theshold and conf threshold
    for image, stat in zip(preds, stats):
        iou = stat[0][:,iou_threshold] # 0.5 iou
        image = image[iou]
        image = image[image[:, 4] > confidence_threshold]
        image = image[:,:4]
        image = xyxy2xywh(image)
        filtered_predictions.append(image)
    
    return filtered_predictions

def calculate_centroid(tensor_xywh):
    # Assuming the tensor has columns: xcentre, ycentre, height, width
    centroid_x = tensor_xywh[:, 0]
    centroid_y = tensor_xywh[:, 1]
    return torch.stack((centroid_x, centroid_y), dim=1)


def pairwise_euclidean_distance(pred_centroids, actual_centroids):
    # Calculate pairwise Euclidean distance between centroids
    pred_centroids = pred_centroids.unsqueeze(1)  # Add a new dimension to pred_centroids
    actual_centroids = actual_centroids.unsqueeze(0)  # Add a new dimension to actual_centroids
    distances = torch.sqrt(torch.sum((pred_centroids - actual_centroids)**2, dim=2))
    
    # Get the minimum distance along each row
    min_distances, _ = distances.min(dim=1)
    return min_distances

def rate_within_threshold(labels, min_distances, threshold):
    
    no_targets = len(labels)
    if len(min_distances) == 0:
        return 0
    else:
        return sum(min_distances < threshold).item() / no_targets

def calculate_tp_fp_fn_within_threshold(labels, min_distances, threshold = 1):
    '''
    Calculate TP as any predicition that has centroid within 1 pixel, FP, FN
    '''
    no_targets = len(labels)
    if len(min_distances) == 0:
        TP = 0
        FP = 0
        FN = no_targets
    else:
        TP = sum(min_distances <= threshold).item()
        FP = sum(min_distances > threshold).item()
        FN = no_targets - TP

    return TP, FP, FN


def calculate_f1_within_threshold(TP, FP, FN):
    eps=1e-16
    recall = TP / (TP + FN + eps)
    precision = TP / (TP + FP + eps)
    f1 = (2 * precision * recall) / (precision + recall + eps)
    return f1


def rmse_tensor(tensor_list):
    
    flattened_tensors = torch.cat([tensor**2 for tensor in tensor_list])
    sum_values = flattened_tensors.sum()

    # Calculate the average
    num_values = flattened_tensors.numel()
    return torch.sqrt(sum_values / num_values).item()


def get_min_euc_rmse_metric(predictions, stats, targets, confidence_threshold, iou_threshold):
    #     get preds above iou and conf threhsold
    filtered_predictions = filter_preds(predictions, stats, confidence_threshold, iou_threshold)
    
    #     for each image, get (min euc distance)**2
    min_euc_dist = []
    fraction_of_preds_within_threshold_1 = []
    fraction_of_preds_within_threshold_05 = []
    min_distances_list = []
    TP_total = 0
    FP_total = 0
    FN_total = 0
    for image_preds, image_targets, stat in zip(filtered_predictions, targets, stats):
        pred_centroids = calculate_centroid(image_preds)
        actual_centroids = calculate_centroid(image_targets)
        min_distances = pairwise_euclidean_distance(pred_centroids, actual_centroids)
        min_distances_list.append(min_distances)
        min_euc_dist.append(min_distances)
        fraction_of_preds_within_threshold_1.append(rate_within_threshold(stat[3], min_distances, threshold = 1))
        fraction_of_preds_within_threshold_05.append(rate_within_threshold(stat[3], min_distances, threshold = 0.5))
        TP, FP, FN = calculate_tp_fp_fn_within_threshold(stat[3], min_distances, threshold = 1)
        TP_total+=TP
        FP_total+=FP
        FN_total+=FN
    #     get rmse of min euc distances
    f1_score = calculate_f1_within_threshold(TP_total, FP_total, FN_total)

    return min_distances_list, rmse_tensor(min_euc_dist), np.mean(fraction_of_preds_within_threshold_1), np.mean(fraction_of_preds_within_threshold_05), f1_score


def plot_euc_threshold_fraction(px, py, threshold, save_dir):
    
    #if isinstance(py, list):
    #    py = torch.tensor(py, dtype=torch.float32, device='cuda:0')

   # px = px.cpu()
    #py = py.cpu()

    # RMSE min euc distance by confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    ax.plot(px, py, linewidth=3, color='blue')
    ax.set_xlabel('Confidence')
    ax.set_ylabel(f'Fraction of objects detected to within {str(threshold)} pixel')
    ax.set_xlim(0, 0.9)
    ax.set_title(f'Fraction of objects detected to within {str(threshold)} pixel')
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)


def plot_euc_rmse(px, py, save_dir):
    
    #if isinstance(py, list):
    #    py = torch.tensor(py, dtype=torch.float32, device='cuda:0')

   # px = px.cpu()
    #py = py.cpu()

    # RMSE min euc distance by confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    ax.plot(px, py, linewidth=3, color='blue')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('RMSE min euclidean distance (excl IOU < 0.5) / pixels')
    ax.set_xlim(0, 0.9)
    ax.set_title('RMSE min euclidean distance by confidence curve')
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)

def plot_f1(px, py, save_dir):
    
    #if isinstance(py, list):
    #    py = torch.tensor(py, dtype=torch.float32, device='cuda:0')

   # px = px.cpu()
    #py = py.cpu()

    # F1 within threshold by confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    ax.plot(px, py, linewidth=3, color='blue')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('F1')
    ax.set_xlim(0, 0.9)
    ax.set_title('F1 by confidence curve, counting TP as predictions within 1 pixel')
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    

def plot_resolution_histogram(min_distances, confidence, save_dir):
    min_distances = [item.item() for tensor in min_distances for item in tensor.view(-1)]
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    ax.hist(min_distances, bins = 60, range = (0,3), color='blue')
    ax.set_xlabel('Pixels')
    ax.set_ylabel('Count of objects detected')
    ax.set_xlim(0, 3.00)
    plt.xticks(np.arange(0,3.00,0.05))
    ax.set_title(f'Distribution of object detection accuracy at F1@1pixel optimum confidence: {confidence}')
    # fig.savefig(save_dir, dpi=250)
    plt.close(fig)

def get_min_euc_rmse_metric_conf_intervals(predictions, stats, targets, iou_threshold, plot = True, save_dir = None):
    confidence_intervals = torch.linspace(0, 1, 100)  # Generate confidence intervals

    min_euc_rmse_metrics = []  # List to store min_euc_rmse_metric for each confidence interval
    fraction_of_preds_within_thresholds_1 = []
    fraction_of_preds_within_thresholds_05 = []
    f1_scores = []
    for confidence_threshold in confidence_intervals:
        # Get min_euc_rmse_metric for the current confidence_threshold
        min_distances_list, min_euc_rmse_metric, fraction_of_preds_within_threshold_1, fraction_of_preds_within_threshold_05, f1_score = get_min_euc_rmse_metric(predictions, stats, targets, confidence_threshold, iou_threshold)
        min_euc_rmse_metrics.append(min_euc_rmse_metric)
        fraction_of_preds_within_thresholds_1.append(fraction_of_preds_within_threshold_1)
        fraction_of_preds_within_thresholds_05.append(fraction_of_preds_within_threshold_05)
        f1_scores.append(f1_score)
        if max(f1_scores) == f1_score:
            optimum_min_distances = min_distances_list
            optimum_confidence = confidence_threshold

    
    if plot:
        plot_euc_rmse(confidence_intervals, min_euc_rmse_metrics, save_dir / f'euc_dist_rmse.jpg')
        plot_euc_threshold_fraction(confidence_intervals, fraction_of_preds_within_thresholds_1, 1, save_dir / f'euc_dist_within_1_pixel_threshold.jpg')
        plot_euc_threshold_fraction(confidence_intervals, fraction_of_preds_within_thresholds_05, 0.5, save_dir / f'euc_dist_within_0.5_pixel_threshold.jpg')
        plot_f1(confidence_intervals, f1_scores, save_dir / f'f1_with_TP_within_1_pixel.jpg')
        plot_resolution_histogram(optimum_min_distances, optimum_confidence, save_dir / f'resolution_histogram.jpg')
    return min_euc_rmse_metrics, f1_scores
