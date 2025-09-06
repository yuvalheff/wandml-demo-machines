import pandas as pd
import numpy as np
import os
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve,
    roc_curve, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from machine_failure_prediction.config import ModelEvalConfig


class ModelEvaluator:
    def __init__(self, config: ModelEvalConfig):
        self.config: ModelEvalConfig = config
        self.app_color_palette = [
            'rgba(99, 110, 250, 0.8)',   # Blue
            'rgba(239, 85, 59, 0.8)',    # Red/Orange-Red
            'rgba(0, 204, 150, 0.8)',    # Green
            'rgba(171, 99, 250, 0.8)',   # Purple
            'rgba(255, 161, 90, 0.8)',   # Orange
            'rgba(25, 211, 243, 0.8)',   # Cyan
            'rgba(255, 102, 146, 0.8)',  # Pink
            'rgba(182, 232, 128, 0.8)',  # Light Green
            'rgba(255, 151, 255, 0.8)',  # Magenta
            'rgba(254, 203, 82, 0.8)'    # Yellow
        ]

    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series, 
                      plots_dir: str) -> Dict[str, Any]:
        """
        Comprehensive model evaluation with plots and metrics
        """
        # Generate predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Generate plots
        os.makedirs(plots_dir, exist_ok=True)
        self._create_plots(y_test, y_pred, y_pred_proba, plots_dir)
        
        return metrics

    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                          y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics"""
        
        # Primary metric (PR-AUC)
        pr_auc = average_precision_score(y_true, y_pred_proba)
        
        # Additional classification metrics
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        return {
            'pr_auc': pr_auc,
            'roc_auc': roc_auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': {
                'true_negative': int(tn),
                'false_positive': int(fp),
                'false_negative': int(fn),
                'true_positive': int(tp)
            },
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }

    def _create_plots(self, y_true: pd.Series, y_pred: np.ndarray, 
                     y_pred_proba: np.ndarray, plots_dir: str):
        """Create comprehensive evaluation plots"""
        
        # 1. Precision-Recall Curve
        self._plot_precision_recall_curve(y_true, y_pred_proba, plots_dir)
        
        # 2. ROC Curve
        self._plot_roc_curve(y_true, y_pred_proba, plots_dir)
        
        # 3. Confusion Matrix
        self._plot_confusion_matrix(y_true, y_pred, plots_dir)
        
        # 4. Probability Distribution
        self._plot_probability_distribution(y_true, y_pred_proba, plots_dir)
        
        # 5. Calibration Plot
        self._plot_calibration_curve(y_true, y_pred_proba, plots_dir)

    def _plot_precision_recall_curve(self, y_true: pd.Series, y_pred_proba: np.ndarray, 
                                    plots_dir: str):
        """Plot Precision-Recall Curve"""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = average_precision_score(y_true, y_pred_proba)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name=f'PR Curve (AUC = {pr_auc:.3f})',
            line=dict(color=self.app_color_palette[0], width=2)
        ))
        
        # Baseline (random classifier)
        baseline = np.sum(y_true) / len(y_true)
        fig.add_hline(y=baseline, line_dash="dash", 
                     annotation_text=f"Random Baseline = {baseline:.3f}",
                     line=dict(color=self.app_color_palette[1], width=1))
        
        fig.update_layout(
            title='Precision-Recall Curve',
            xaxis_title='Recall',
            yaxis_title='Precision',
            **self._get_plot_style()
        )
        
        self._save_plot(fig, os.path.join(plots_dir, 'precision_recall_curve.html'))

    def _plot_roc_curve(self, y_true: pd.Series, y_pred_proba: np.ndarray, 
                       plots_dir: str):
        """Plot ROC Curve"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.3f})',
            line=dict(color=self.app_color_palette[0], width=2)
        ))
        
        # Diagonal line (random classifier)
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Baseline',
            line=dict(color=self.app_color_palette[1], width=1, dash='dash')
        ))
        
        fig.update_layout(
            title='Receiver Operating Characteristic (ROC) Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            **self._get_plot_style()
        )
        
        self._save_plot(fig, os.path.join(plots_dir, 'roc_curve.html'))

    def _plot_confusion_matrix(self, y_true: pd.Series, y_pred: np.ndarray, 
                              plots_dir: str):
        """Plot Confusion Matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted No Failure', 'Predicted Failure'],
            y=['Actual No Failure', 'Actual Failure'],
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Confusion Matrix',
            **self._get_plot_style()
        )
        
        self._save_plot(fig, os.path.join(plots_dir, 'confusion_matrix.html'))

    def _plot_probability_distribution(self, y_true: pd.Series, y_pred_proba: np.ndarray, 
                                     plots_dir: str):
        """Plot probability distribution by class"""
        df = pd.DataFrame({
            'probability': y_pred_proba,
            'actual_class': y_true.map({0: 'No Failure', 1: 'Failure'})
        })
        
        fig = go.Figure()
        
        for i, class_name in enumerate(['No Failure', 'Failure']):
            class_data = df[df['actual_class'] == class_name]['probability']
            fig.add_trace(go.Histogram(
                x=class_data,
                name=class_name,
                opacity=0.7,
                nbinsx=30,
                marker_color=self.app_color_palette[i]
            ))
        
        fig.update_layout(
            title='Predicted Probability Distribution by Actual Class',
            xaxis_title='Predicted Probability of Failure',
            yaxis_title='Count',
            barmode='overlay',
            **self._get_plot_style()
        )
        
        self._save_plot(fig, os.path.join(plots_dir, 'probability_distribution.html'))

    def _plot_calibration_curve(self, y_true: pd.Series, y_pred_proba: np.ndarray, 
                               plots_dir: str):
        """Plot calibration curve"""
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=10
        )
        
        fig = go.Figure()
        
        # Calibration curve
        fig.add_trace(go.Scatter(
            x=mean_predicted_value,
            y=fraction_of_positives,
            mode='lines+markers',
            name='Model',
            line=dict(color=self.app_color_palette[0], width=2),
            marker=dict(size=8)
        ))
        
        # Perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(color=self.app_color_palette[1], width=1, dash='dash')
        ))
        
        fig.update_layout(
            title='Calibration Plot (Reliability Curve)',
            xaxis_title='Mean Predicted Probability',
            yaxis_title='Fraction of Positives',
            **self._get_plot_style()
        )
        
        self._save_plot(fig, os.path.join(plots_dir, 'calibration_curve.html'))

    def _get_plot_style(self) -> Dict[str, Any]:
        """Get consistent plot styling"""
        return {
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'font': dict(color='#8B5CF6', size=12),
            'title_font': dict(color='#7C3AED', size=16),
            'xaxis': dict(
                gridcolor='rgba(139,92,246,0.2)',
                zerolinecolor='rgba(139,92,246,0.3)',
                tickfont=dict(color='#8B5CF6', size=11),
                title_font=dict(color='#7C3AED', size=12)
            ),
            'yaxis': dict(
                gridcolor='rgba(139,92,246,0.2)',
                zerolinecolor='rgba(139,92,246,0.3)',
                tickfont=dict(color='#8B5CF6', size=11),
                title_font=dict(color='#7C3AED', size=12)
            ),
            'legend': dict(font=dict(color='#8B5CF6', size=11))
        }

    def _save_plot(self, fig, filepath: str):
        """Save plot with consistent configuration"""
        fig.write_html(
            filepath, 
            include_plotlyjs=True, 
            config={'responsive': True, 'displayModeBar': False}
        )
