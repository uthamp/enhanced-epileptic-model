import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix
from sklearn.model_selection import learning_curve
import warnings
warnings.filterwarnings('ignore')

def load_model_package(file_path):
    """Load model and associated metrics from pickle file."""
    with open(file_path, 'rb') as file:
        model_package = pickle.load(file)
    return model_package

def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.Blues):
    """Plot confusion matrix as a heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true, scores, title='ROC Curve'):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
    
    return roc_auc

def plot_precision_recall_curve(y_true, scores, title='Precision-Recall Curve'):
    """Plot precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, scores)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()
    
    return pr_auc

def plot_cross_validation_results(cv_metrics, title='Cross-Validation Results'):
    """Plot cross-validation results."""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
    metric_values = [cv_metrics[m] for m in metrics]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, metric_values, color='steelblue')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.ylim(0, 1.1)
    plt.ylabel('Score')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_learning_curve(model, X, y, title='Learning Curve', cv=5):
    """Plot learning curve to show model performance with increasing data."""
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy')
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, 
             label='Training accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    
    plt.plot(train_sizes, test_mean, color='green', marker='s', markersize=5, 
             label='Validation accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    
    plt.xlabel('Training Size')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_feature_importance(model, feature_names, top_n=20):
    """Visualize feature importance for models that support it."""
    if hasattr(model, 'feature_importances_'):
        # Get feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot the top N features
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importances')
        plt.bar(range(top_n), importances[indices][:top_n], align='center')
        plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=90)
        plt.tight_layout()
        plt.show()
    else:
        print("This model doesn't provide feature importances directly.")

def evaluate_model_package(model_package, X_test, y_test=None):
    """Comprehensive evaluation of a model package."""
    model = model_package['model']
    metrics = model_package['performance_metrics']
    cv_metrics = model_package['cross_validation_metrics']
    training_info = model_package['training_info']
    
    print("====== MODEL EVALUATION REPORT ======")
    print(f"\nModel Type: {training_info['model_type']}")
    print(f"Dataset Size: {training_info['dataset_size']} samples")
    print(f"Class Distribution: {training_info['class_distribution']}")
    print(f"Training Timestamp: {training_info['timestamp']}")
    
    print("\n----- Performance Metrics -----")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    if 'auc' in metrics and not np.isnan(metrics['auc']):
        print(f"AUC: {metrics['auc']:.4f}")
    
    print("\n----- Cross-Validation Metrics -----")
    print(f"CV Accuracy: {cv_metrics['accuracy']:.4f}")
    print(f"CV Precision: {cv_metrics['precision']:.4f}")
    print(f"CV Recall: {cv_metrics['recall']:.4f}")
    print(f"CV F1 Score: {cv_metrics['f1_score']:.4f}")
    if 'auc' in cv_metrics and not np.isnan(cv_metrics['auc']):
        print(f"CV AUC: {cv_metrics['auc']:.4f}")
    
    # If we have test data, generate visuals
    if X_test is not None:
        print("\n----- Generating Visualizations -----")
        
        # Get predictions
        predictions = model.predict(X_test)
        
        # For anomaly detection with no true labels
        if y_test is None or len(np.unique(y_test)) <= 1:
            # Get anomaly scores
            anomaly_scores = model.get_anomaly_scores(X_test)
            threshold = np.percentile(anomaly_scores, 90)
            simulated_labels = np.where(anomaly_scores > threshold, 1, 0)
            
            # Plot confusion matrix
            cm = confusion_matrix(simulated_labels, predictions)
            plot_confusion_matrix(cm, title='Confusion Matrix (Simulated Labels)')
            
            # ROC curve
            try:
                plot_roc_curve(simulated_labels, anomaly_scores, 
                             title='ROC Curve (Simulated Labels)')
            except:
                print("Could not generate ROC curve.")
            
            # Precision-Recall curve
            try:
                plot_precision_recall_curve(simulated_labels, anomaly_scores, 
                                         title='Precision-Recall Curve (Simulated Labels)')
            except:
                print("Could not generate Precision-Recall curve.")
        
        # For supervised learning with true labels
        else:
            # Get probability scores if available
            if hasattr(model, 'predict_proba'):
                try:
                    prob_scores = model.predict_proba(X_test)[:, 1]
                except:
                    # If multi-class or other issues
                    prob_scores = predictions.astype(float)
            else:
                prob_scores = predictions.astype(float)
            
            # Plot confusion matrix
            cm = confusion_matrix(y_test, predictions)
            plot_confusion_matrix(cm, title='Confusion Matrix')
            
            # ROC curve
            try:
                plot_roc_curve(y_test, prob_scores, title='ROC Curve')
            except:
                print("Could not generate ROC curve.")
            
            # Precision-Recall curve
            try:
                plot_precision_recall_curve(y_test, prob_scores, title='Precision-Recall Curve')
            except:
                print("Could not generate Precision-Recall curve.")
        
        # Cross-validation results
        plot_cross_validation_results(cv_metrics)
    
    return {
        'model': model,
        'metrics': metrics,
        'cv_metrics': cv_metrics,
        'training_info': training_info
    }

# Example usage
if __name__ == "__main__":
    # Load the model package
    model_package = load_model_package('comprehensive_epilepsy_model.pkl')
    
    # Load test data
    dataset = pd.read_csv('chbmit_preprocessed_data.csv')
    dataset.dropna(inplace=True)
    
    # Separate features and target
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    
    # Split data
    from sklearn.model_selection import train_test_split
    _, X_test = train_test_split(X, test_size=0.2, random_state=42)
    
    # Evaluate the model
    evaluate_model_package(model_package, X_test)