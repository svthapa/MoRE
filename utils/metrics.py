from sklearn.metrics import f1_score,roc_auc_score, precision_recall_curve, average_precision_score, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt 

def calculate_multilabel_map(y_true, y_score):
    """
    Calculate Mean Average Precision (mAP) for multilabel classification.
    
    Args:
        y_true (array-like, shape (n_samples, n_classes)): True labels (multilabel format).
        y_score (array-like, shape (n_samples, n_classes)): Predicted probabilities.

    Returns:
        float: Mean Average Precision (mAP) score.
    """
    num_classes = y_true.shape[1]
    ap_scores = []

    for label in range(num_classes):
        ap = average_precision_score(y_true[:, label], y_score[:, label])
        ap_scores.append(ap)

    mAP = sum(ap_scores) / len(ap_scores)
    return mAP

def calc_roc_auc(all_labels, all_probs):
    # Calculate ROC AUC for each label
    num_labels = all_labels.shape[1]
    roc_auc_scores = []
    for i in range(num_labels):
        # if any(all_labels[:, i]):  # Check if there are any positive samples for this label
        roc_auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
        # else:
        #     roc_auc = 0.0  # Assign a default ROC AUC of 0 if there are no positive samples
        roc_auc_scores.append(roc_auc)

    # Calculate macro and micro averaged ROC AUC
    roc_auc_macro = np.mean(roc_auc_scores)
    roc_auc_micro = roc_auc_score(all_labels.ravel(), all_probs.ravel())

    return roc_auc_macro, roc_auc_micro

def plot_roc_curve(all_labels, all_probs, num_labels, save_path='roc_curve.png'):
    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))
    
    # Calculate and plot ROC curve for each label
    for i in range(num_labels):
        fpr, tpr, _ = roc_curve(all_labels[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Label {i} (area = {roc_auc:.2f})')
        
    # Calculate and plot micro-average ROC curve
    fpr, tpr, _ = roc_curve(all_labels.ravel(), all_probs.ravel())
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='deeppink', lw=2, linestyle=':', label=f'Micro-average ROC curve (area = {roc_auc:.2f})')
    
    # Plot the random chance line
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # Add labels and legend
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    # Save the plot
    plt.savefig(save_path)
    plt.close()
    
def plot_precision_recall_curve(y_true, y_probs, save_path):
    n_classes = y_true.shape[1]
    plt.figure(figsize=(10, 8))
    
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_probs[:, i])
        avg_precision = average_precision_score(y_true[:, i], y_probs[:, i])
        plt.plot(recall, precision, lw=2, label=f'Label {i} (area = {avg_precision:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc='best')
    plt.grid(alpha=0.2)
    plt.savefig(save_path)
    plt.close()
    
def precision_recall_per_label(true_labels, predictions):
    """
    Calculate precision and recall for each label.
    
    Parameters:
    - true_labels (numpy array): Ground truth labels.
    - predictions (numpy array): Predicted labels.
    
    Returns:
    - precisions (list): List of precisions for each label.
    - recalls (list): List of recalls for each label.
    """
    
    # Check if dimensions match
    if true_labels.shape != predictions.shape:
        raise ValueError("Dimensions of true labels and predictions do not match.")
    
    num_labels = true_labels.shape[1]
    precisions = []
    recalls = []
    for i in range(num_labels):
        TP = np.sum((true_labels[:, i] == 1) & (predictions[:, i] == 1))
        FP = np.sum((true_labels[:, i] == 0) & (predictions[:, i] == 1))
        FN = np.sum((true_labels[:, i] == 1) & (predictions[:, i] == 0))
        
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    return precisions, recalls

def accuracy_per_label(true_labels, predictions):
    """
    Calculate accuracy for each label.

    Parameters:
    - true_labels (numpy array): Ground truth labels.
    - predictions (numpy array): Predicted labels.

    Returns:
    - accuracies (list): List of accuracies for each label.
    """

    # Check if dimensions match
    if true_labels.shape != predictions.shape:
        raise ValueError("Dimensions of true labels and predictions do not match.")

    # Calculate accuracies
    num_labels = true_labels.shape[1]
    accuracies = []
    for i in range(num_labels):
        correct_predictions = np.sum(true_labels[:, i] == predictions[:, i])
        total_instances = true_labels.shape[0]
        accuracy = correct_predictions / total_instances
        accuracies.append(accuracy)

    return accuracies

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0