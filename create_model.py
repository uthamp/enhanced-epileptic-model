import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC, OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM, Input, concatenate

# Suppress warnings
warnings.filterwarnings('ignore')

class ComprehensiveEpilepsyModel:
    """
    Comprehensive model that combines anomaly detection and deep learning approaches
    for epilepsy seizure detection and prediction.
    """
    
    def __init__(self):
        # Initialize all models
        self.initialize_anomaly_models()
        self.initialize_classical_models()
        self.initialize_deep_learning_models()
        
        # Flags for model state
        self.anomaly_models_fitted = False
        self.classical_models_fitted = False
        self.deep_learning_models_fitted = False
        self.feature_extractor_fitted = False
        
        # Thresholds for anomaly detection
        self.anomaly_thresholds = {}
        
        # Feature selector
        self.feature_selector = None
        
        # Scaler for standardizing data
        self.scaler = StandardScaler()
        
    def initialize_anomaly_models(self):
        """Initialize anomaly detection models."""
        self.iso_forest = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42
        )
        
        self.one_class_svm = OneClassSVM(
            nu=0.1,
            kernel="rbf",
            gamma='scale'
        )
        
        self.lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.1,
            novelty=True
        )
    
    def initialize_classical_models(self):
        """Initialize classical machine learning models."""
        self.rf_clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.svm_clf = SVC(
            kernel='rbf',
            probability=True,
            class_weight='balanced',
            random_state=42
        )
        
        self.xgb_clf = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    
    def initialize_deep_learning_models(self):
        """Initialize deep learning models for feature extraction and classification."""
        # CNN feature extractor
        self.cnn_model = None
        
        # LSTM feature extractor
        self.lstm_model = None
        
        # Combined CNN-LSTM feature extractor
        self.cnn_lstm_model = None
    
    def create_cnn_model(self, input_shape):
        """Create a CNN model for feature extraction."""
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=128, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=128, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(100, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def create_lstm_model(self, input_shape):
        """Create an LSTM model for feature extraction."""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(100, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def create_cnn_lstm_model(self, input_shape):
        """Create a combined CNN-LSTM model for feature extraction."""
        # CNN branch
        cnn_input = Input(shape=input_shape)
        cnn = Conv1D(filters=64, kernel_size=3, activation='relu')(cnn_input)
        cnn = MaxPooling1D(pool_size=2)(cnn)
        cnn = Conv1D(filters=128, kernel_size=3, activation='relu')(cnn)
        cnn = MaxPooling1D(pool_size=2)(cnn)
        cnn = Flatten()(cnn)
        
        # LSTM branch
        lstm_input = Input(shape=input_shape)
        lstm = LSTM(50, return_sequences=True)(lstm_input)
        lstm = Dropout(0.2)(lstm)
        lstm = LSTM(50, return_sequences=False)(lstm)
        
        # Combine branches
        combined = concatenate([cnn, lstm])
        combined = Dense(100, activation='relu')(combined)
        output = Dense(1, activation='sigmoid')(combined)
        
        model = Model(inputs=[cnn_input, lstm_input], outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def prepare_data_for_deep_learning(self, X):
        """
        Reshape the data for deep learning models.
        Assuming X is a 2D array of shape (n_samples, n_features).
        """
        # Reshape to (n_samples, n_timesteps, n_features)
        # Here we're treating each feature as a timestep for a single feature
        n_samples = X.shape[0]
        n_timesteps = X.shape[1]
        n_features = 1
        return X.reshape(n_samples, n_timesteps, n_features)
    
    def fit_anomaly_models(self, X):
        """Fit anomaly detection models."""
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit the models
        self.iso_forest.fit(X_scaled)
        self.one_class_svm.fit(X_scaled)
        self.lof.fit(X_scaled)
        
        # Calculate and store anomaly scores for threshold setting
        self.iso_forest_scores = self.iso_forest.score_samples(X_scaled)
        self.one_class_svm_scores = self.one_class_svm.score_samples(X_scaled)
        self.lof_scores = self.lof.score_samples(X_scaled)
        
        # Set thresholds at the 10th percentile
        self.anomaly_thresholds['iso_forest'] = np.percentile(self.iso_forest_scores, 10)
        self.anomaly_thresholds['one_class_svm'] = np.percentile(self.one_class_svm_scores, 10)
        self.anomaly_thresholds['lof'] = np.percentile(self.lof_scores, 10)
        
        self.anomaly_models_fitted = True
    
    def create_feature_extractor(self, X):
        """Create and train an autoencoder for feature extraction when only normal data is available."""
        # Reshape data for CNNs and LSTMs
        X_reshaped = self.prepare_data_for_deep_learning(X)
        
        # Create the CNN model
        self.cnn_model = self.create_cnn_model(input_shape=(X_reshaped.shape[1], X_reshaped.shape[2]))
        
        # Create the LSTM model
        self.lstm_model = self.create_lstm_model(input_shape=(X_reshaped.shape[1], X_reshaped.shape[2]))
        
        # For demonstration purposes, we won't actually train these models without seizure data
        # In a real scenario, we could use self-supervised learning techniques
        
        self.feature_extractor_fitted = True
    
    def fit(self, X, y=None):
        """Fit the model to the data."""
        # Check if we have seizure examples
        if y is not None and len(np.unique(y)) > 1:
            print("Training with multi-class data (normal and seizure examples)...")
            self.fit_with_seizure_data(X, y)
        else:
            print("Training with only normal data (anomaly detection approach)...")
            self.fit_without_seizure_data(X)
    
    def fit_without_seizure_data(self, X):
        """Fit models using anomaly detection when only normal data is available."""
        # Fit anomaly detection models
        self.fit_anomaly_models(X)
        
        # Create feature extractors for future use
        self.create_feature_extractor(X)
        
        print("Anomaly detection models successfully trained.")
        print("Feature extractors prepared for future use with seizure data.")
    
    def fit_with_seizure_data(self, X, y):
        """Fit models when both normal and seizure data are available."""
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Feature selection
        self.feature_selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
        X_selected = self.feature_selector.fit_transform(X_scaled, y)
        
        # Train classical models
        self.rf_clf.fit(X_selected, y)
        self.svm_clf.fit(X_selected, y)
        self.xgb_clf.fit(X_selected, y)
        self.classical_models_fitted = True
        
        # Train deep learning models
        X_reshaped = self.prepare_data_for_deep_learning(X_scaled)
        
        # Create and train CNN
        self.cnn_model = self.create_cnn_model(input_shape=(X_reshaped.shape[1], X_reshaped.shape[2]))
        self.cnn_model.fit(X_reshaped, y, epochs=10, batch_size=32, verbose=0)
        
        # Create and train LSTM
        self.lstm_model = self.create_lstm_model(input_shape=(X_reshaped.shape[1], X_reshaped.shape[2]))
        self.lstm_model.fit(X_reshaped, y, epochs=10, batch_size=32, verbose=0)
        
        self.deep_learning_models_fitted = True
        
        print("Classical and deep learning models successfully trained.")
    
    def predict_anomaly(self, X):
        """Make predictions using anomaly detection models."""
        if not self.anomaly_models_fitted:
            raise ValueError("Anomaly models not fitted yet. Call fit() first.")
        
        # Scale the data
        X_scaled = self.scaler.transform(X)
        
        # Get anomaly scores
        iso_scores = self.iso_forest.score_samples(X_scaled)
        svm_scores = self.one_class_svm.score_samples(X_scaled)
        lof_scores = self.lof.score_samples(X_scaled)
        
        # Make individual predictions (1 for normal, -1 for anomaly/seizure)
        iso_preds = np.where(iso_scores < self.anomaly_thresholds['iso_forest'], -1, 1)
        svm_preds = np.where(svm_scores < self.anomaly_thresholds['one_class_svm'], -1, 1)
        lof_preds = np.where(lof_scores < self.anomaly_thresholds['lof'], -1, 1)
        
        # Ensemble voting (if 2 or more models predict anomaly, classify as anomaly)
        results = []
        for i in range(len(iso_preds)):
            votes = [iso_preds[i], svm_preds[i], lof_preds[i]]
            # Count votes for anomaly (-1)
            anomaly_votes = votes.count(-1)
            # If majority of models predict anomaly, classify as anomaly
            if anomaly_votes >= 2:
                results.append(-1)  # Anomaly (potential seizure)
            else:
                results.append(1)   # Normal
        
        # Convert from -1/1 to 0/1 format (0: normal, 1: seizure/anomaly)
        # This matches the expected output format where 1 indicates seizure
        final_results = [1 if res == -1 else 0 for res in results]
        return np.array(final_results)
    
    def predict_with_seizure_data(self, X):
        """Make predictions using models trained on both normal and seizure data."""
        if not self.classical_models_fitted:
            raise ValueError("Classical models not fitted yet. Call fit() with labeled data first.")
        
        # Scale the data
        X_scaled = self.scaler.transform(X)
        
        # Feature selection
        X_selected = self.feature_selector.transform(X_scaled)
        
        # Get predictions from classical models
        rf_pred = self.rf_clf.predict(X_selected)
        svm_pred = self.svm_clf.predict(X_selected)
        xgb_pred = self.xgb_clf.predict(X_selected)
        
        # Reshape for deep learning
        X_reshaped = self.prepare_data_for_deep_learning(X_scaled)
        
        # Get predictions from deep learning models
        cnn_pred = (self.cnn_model.predict(X_reshaped) > 0.5).astype(int).flatten()
        lstm_pred = (self.lstm_model.predict(X_reshaped) > 0.5).astype(int).flatten()
        
        # Ensemble voting
        ensemble_pred = np.zeros(len(rf_pred))
        for i in range(len(rf_pred)):
            votes = [rf_pred[i], svm_pred[i], xgb_pred[i], cnn_pred[i], lstm_pred[i]]
            # Count votes for seizure (1)
            seizure_votes = votes.count(1)
            # If majority of models predict seizure, classify as seizure
            if seizure_votes >= 3:
                ensemble_pred[i] = 1
        
        return ensemble_pred
    
    def predict(self, X):
        """Make predictions based on available trained models."""
        if self.classical_models_fitted and self.deep_learning_models_fitted:
            return self.predict_with_seizure_data(X)
        elif self.anomaly_models_fitted:
            return self.predict_anomaly(X)
        else:
            raise ValueError("No models fitted yet. Call fit() first.")
            
    def calculate_accuracy_metrics(self, X_test, y_test=None):
        """
        Calculate accuracy metrics for the model.
        If y_test is None, uses anomaly detection approach with simulated labels.
        """
        if y_test is None or len(np.unique(y_test)) <= 1:
            return self.calculate_anomaly_detection_metrics(X_test)
        else:
            return self.calculate_supervised_metrics(X_test, y_test)
    
    def calculate_anomaly_detection_metrics(self, X_test):
        """Calculate metrics for anomaly detection approach."""
        # Get anomaly scores and predictions
        anomaly_scores = self.get_anomaly_scores(X_test)
        
        # Use top 10% most anomalous as simulated positives
        threshold = np.percentile(anomaly_scores, 90)
        simulated_labels = np.where(anomaly_scores > threshold, 1, 0)
        
        # Get model predictions
        predictions = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(simulated_labels, predictions)
        precision = precision_score(simulated_labels, predictions, zero_division=0)
        recall = recall_score(simulated_labels, predictions, zero_division=0)
        f1 = f1_score(simulated_labels, predictions, zero_division=0)
        
        # ROC curve and AUC (using anomaly scores directly)
        try:
            auc_score = roc_auc_score(simulated_labels, anomaly_scores)
        except:
            auc_score = np.nan
            
        # Confusion matrix
        cm = confusion_matrix(simulated_labels, predictions)
        
        # Return all metrics as a dictionary
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc_score,
            'confusion_matrix': cm,
            'method': 'anomaly_detection',
            'note': 'These metrics are simulated as no true seizure labels are available'
        }
        
        return metrics
    
    def calculate_supervised_metrics(self, X_test, y_test):
        """Calculate metrics for supervised learning approach."""
        # Get predictions
        predictions = self.predict(X_test)
        
        # For probability scores, use a blend of all models
        if self.classical_models_fitted:
            # Scale data
            X_scaled = self.scaler.transform(X_test)
            X_selected = self.feature_selector.transform(X_scaled)
            
            # Get probabilities from each model
            prob_scores = (
                self.rf_clf.predict_proba(X_selected)[:, 1] + 
                self.svm_clf.predict_proba(X_selected)[:, 1] + 
                self.xgb_clf.predict_proba(X_selected)[:, 1]
            ) / 3
        else:
            # If classical models aren't available, use dummy scores
            prob_scores = predictions.astype(float)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        
        # ROC curve and AUC
        try:
            auc_score = roc_auc_score(y_test, prob_scores)
        except:
            auc_score = np.nan
            
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        
        # Return all metrics as a dictionary
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc_score,
            'confusion_matrix': cm,
            'method': 'supervised_learning'
        }
        
        return metrics
        
    def cross_validate(self, X, y=None, cv=5):
        """
        Perform cross-validation to get robust performance metrics.
        If y is None, uses anomaly detection approach with simulated labels.
        """
        if y is None or len(np.unique(y)) <= 1:
            return self.cross_validate_anomaly(X, cv)
        else:
            return self.cross_validate_supervised(X, y, cv)
    
    def cross_validate_anomaly(self, X, cv=5):
        """Cross-validate anomaly detection approach."""
        # Split data into folds
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Metrics for each fold
        metrics_list = []
        
        for train_idx, test_idx in kf.split(X):
            X_train_fold, X_test_fold = X[train_idx], X[test_idx]
            
            # Create a new model and fit it
            fold_model = ComprehensiveEpilepsyModel()
            fold_model.fit(X_train_fold)
            
            # Calculate metrics
            fold_metrics = fold_model.calculate_anomaly_detection_metrics(X_test_fold)
            metrics_list.append(fold_metrics)
        
        # Calculate average metrics
        avg_metrics = {
            'accuracy': np.mean([m['accuracy'] for m in metrics_list]),
            'precision': np.mean([m['precision'] for m in metrics_list]),
            'recall': np.mean([m['recall'] for m in metrics_list]),
            'f1_score': np.mean([m['f1_score'] for m in metrics_list]),
            'auc': np.mean([m['auc'] for m in metrics_list if not np.isnan(m['auc'])]),
            'method': 'anomaly_detection',
            'note': 'Cross-validated metrics using simulated labels',
            'fold_metrics': metrics_list
        }
        
        return avg_metrics
    
    def cross_validate_supervised(self, X, y, cv=5):
        """Cross-validate supervised learning approach."""
        # Split data into folds
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Metrics for each fold
        metrics_list = []
        
        for train_idx, test_idx in skf.split(X, y):
            X_train_fold, X_test_fold = X[train_idx], X[test_idx]
            y_train_fold, y_test_fold = y[train_idx], y[test_idx]
            
            # Create a new model and fit it
            fold_model = ComprehensiveEpilepsyModel()
            fold_model.fit(X_train_fold, y_train_fold)
            
            # Calculate metrics
            fold_metrics = fold_model.calculate_supervised_metrics(X_test_fold, y_test_fold)
            metrics_list.append(fold_metrics)
        
        # Calculate average metrics
        avg_metrics = {
            'accuracy': np.mean([m['accuracy'] for m in metrics_list]),
            'precision': np.mean([m['precision'] for m in metrics_list]),
            'recall': np.mean([m['recall'] for m in metrics_list]),
            'f1_score': np.mean([m['f1_score'] for m in metrics_list]),
            'auc': np.mean([m['auc'] for m in metrics_list if not np.isnan(m['auc'])]),
            'method': 'supervised_learning',
            'note': 'Cross-validated metrics',
            'fold_metrics': metrics_list
        }
        
        return avg_metrics
    
    def get_anomaly_scores(self, X):
        """Get normalized anomaly scores."""
        if not self.anomaly_models_fitted:
            raise ValueError("Anomaly models not fitted yet. Call fit() first.")
        
        # Scale the data
        X_scaled = self.scaler.transform(X)
        
        # Get anomaly scores
        iso_scores = self.iso_forest.score_samples(X_scaled)
        svm_scores = self.one_class_svm.score_samples(X_scaled)
        lof_scores = self.lof.score_samples(X_scaled)
        
        # Normalize scores to 0-1 range where higher values indicate more anomalous
        iso_scores_norm = 1 - (iso_scores - np.min(self.iso_forest_scores)) / (np.max(self.iso_forest_scores) - np.min(self.iso_forest_scores))
        svm_scores_norm = 1 - (svm_scores - np.min(self.one_class_svm_scores)) / (np.max(self.one_class_svm_scores) - np.min(self.one_class_svm_scores))
        lof_scores_norm = 1 - (lof_scores - np.min(self.lof_scores)) / (np.max(self.lof_scores) - np.min(self.lof_scores))
        
        # Average the normalized scores
        ensemble_scores = (iso_scores_norm + svm_scores_norm + lof_scores_norm) / 3
        
        return ensemble_scores

# Usage Example
def main():
    # Load data
    print("Loading dataset...")
    dataset = pd.read_csv('chbmit_preprocessed_data.csv')
    
    # Handle missing values
    dataset.dropna(inplace=True)
    
    # Separate features and target
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    
    # Check class distribution
    class_counts = np.bincount(y)
    print(f"Class distribution in dataset: {class_counts}")
    
    # Split data
    if len(class_counts) == 1:
        # For anomaly detection, just split X
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
        y_train, y_test = None, None
    else:
        # For supervised learning, stratify the split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create and train model
    model = ComprehensiveEpilepsyModel()
    
    # If only one class is present (all normal or all seizure)
    if len(class_counts) == 1:
        print("Only one class detected, using anomaly detection approach...")
        model.fit(X_train)
        
        # Calculate detailed performance metrics
        print("\nCalculating model performance metrics...")
        metrics = model.calculate_accuracy_metrics(X_test)
        
        print(f"\nModel Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        if not np.isnan(metrics['auc']):
            print(f"AUC: {metrics['auc']:.4f}")
        
        print("\nConfusion Matrix (simulated):")
        print(metrics['confusion_matrix'])
        
        # Cross-validation
        print("\nPerforming cross-validation...")
        cv_metrics = model.cross_validate(X, cv=5)
        
        print(f"\nCross-validated Accuracy: {cv_metrics['accuracy']:.4f}")
        print(f"Cross-validated Precision: {cv_metrics['precision']:.4f}")
        print(f"Cross-validated Recall: {cv_metrics['recall']:.4f}")
        print(f"Cross-validated F1 Score: {cv_metrics['f1_score']:.4f}")
        if not np.isnan(cv_metrics['auc']):
            print(f"Cross-validated AUC: {cv_metrics['auc']:.4f}")
        
        print("\nNote: These metrics are based on simulated anomaly labels since no true seizure data is available.")
        
    # If both classes are present
    else:
        print("Both classes detected, using supervised learning approach...")
        model.fit(X_train, y_train)
        
        # Calculate detailed performance metrics
        print("\nCalculating model performance metrics...")
        metrics = model.calculate_accuracy_metrics(X_test, y_test)
        
        print(f"\nModel Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        if not np.isnan(metrics['auc']):
            print(f"AUC: {metrics['auc']:.4f}")
        
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        
        print("\nClassification Report:")
        print(classification_report(y_test, model.predict(X_test)))
        
        # Cross-validation
        print("\nPerforming cross-validation...")
        cv_metrics = model.cross_validate(X, y, cv=5)
        
        print(f"\nCross-validated Accuracy: {cv_metrics['accuracy']:.4f}")
        print(f"Cross-validated Precision: {cv_metrics['precision']:.4f}")
        print(f"Cross-validated Recall: {cv_metrics['recall']:.4f}")
        print(f"Cross-validated F1 Score: {cv_metrics['f1_score']:.4f}")
        if not np.isnan(cv_metrics['auc']):
            print(f"Cross-validated AUC: {cv_metrics['auc']:.4f}")
    
    # Save model
    print("\nSaving model with accuracy metrics...")
    model_package = {
        'model': model,
        'performance_metrics': metrics,
        'cross_validation_metrics': cv_metrics,
        'training_info': {
            'dataset_size': len(dataset),
            'class_distribution': class_counts.tolist(),
            'model_type': 'anomaly_detection' if len(class_counts) == 1 else 'supervised_learning',
            'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    
    with open('comprehensive_epilepsy_model.pkl', 'wb') as file:
        pickle.dump(model_package, file)
    print("Model successfully saved as comprehensive_epilepsy_model.pkl")

if __name__ == "__main__":
    main()