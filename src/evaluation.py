"""
Evaluación: métricas, visualización, análisis
"""
import torch
import torch.nn as nn
import numpy as np
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from config import PLOT_DIR, RESULT_DIR

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evalúa rendimiento del modelo"""

    def __init__(self, model, device='cpu', is_autoencoder=False):
        """
        Args:
            model: modelo a evaluar
            device: dispositivo
            is_autoencoder: si True, evalúa reconstrucción X→X̂ (comparar contra X_batch).
                            si False, evalúa predicción aerodinámica (comparar contra Y_batch).
        """
        self.model = model
        self.device = device
        self.is_autoencoder = is_autoencoder

    @torch.no_grad()
    def evaluate(self, test_loader, return_predictions=False):
        """
        Evalúa el modelo en dataset de test.

        Returns:
            dict con 'metrics' y, opcionalmente, 'y_true', 'y_pred', 'z'.
        """
        self.model.eval()

        y_true_list = []
        y_pred_list = []
        z_list = []

        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(self.device)

            # Predicción
            output = self.model(X_batch)

            if isinstance(output, tuple):
                # AE devuelve (x_recon, z)
                y_pred = output[0]
                z_list.append(output[1].cpu().numpy())
            elif isinstance(output, dict):
                y_pred = output.get('shock_prob', output.get('latent'))
                if 'latent' in output:
                    z_list.append(output['latent'].cpu().numpy())
            else:
                y_pred = output

            # Target: X para reconstrucción, Y para predicción aerodinámica
            if self.is_autoencoder:
                y_true_list.append(X_batch.cpu().numpy())
            else:
                y_true_list.append(Y_batch.cpu().numpy())

            y_pred_list.append(y_pred.cpu().numpy())

        y_true = np.concatenate(y_true_list, axis=0)
        y_pred = np.concatenate(y_pred_list, axis=0)

        # Asegurar mismas dimensiones (recortar si difieren)
        min_cols = min(y_true.shape[1], y_pred.shape[1])
        y_true = y_true[:, :min_cols]
        y_pred = y_pred[:, :min_cols]

        metrics = {
            'mse': float(mean_squared_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae': float(mean_absolute_error(y_true, y_pred)),
        }

        # R² por canal
        output_names = ['Cp', 'Cfx', 'Cfy', 'Cfz'] if not self.is_autoencoder else None
        for i in range(min_cols):
            name = output_names[i] if (output_names and i < len(output_names)) else f'feat_{i}'
            metrics[f'r2_{name}'] = float(r2_score(y_true[:, i], y_pred[:, i]))

        result = {'metrics': metrics}

        if return_predictions:
            result['y_true'] = y_true
            result['y_pred'] = y_pred
            if z_list:
                result['z'] = np.concatenate(z_list, axis=0)

        return result

    def log_metrics(self, metrics):
        logger.info("=" * 60)
        logger.info("EVALUATION METRICS")
        logger.info("=" * 60)
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"{key:20s}: {value:.6f}")
        logger.info("=" * 60)



class VisualizationTools:
    """Herramientas de visualización"""
    
    @staticmethod
    def plot_losses(train_losses, val_losses, save_path=None):
        """Grafica curvas de loss"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss', linewidth=2)
        plt.plot(val_losses, label='Val Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title('Training and Validation Loss')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        return plt.gcf()
    
    @staticmethod
    def plot_predictions_vs_truth(y_true, y_pred, feature_names=None, save_path=None):
        """Grafica predicciones vs valores verdaderos"""
        if feature_names is None:
            feature_names = ['Cp', 'Cfx', 'Cfy', 'Cfz']
        
        n_features = y_true.shape[1]
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i in range(min(n_features, 4)):
            ax = axes[i]
            
            ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.3, s=1)
            
            # Línea de referencia
            min_val = min(y_true[:, i].min(), y_pred[:, i].min())
            max_val = max(y_true[:, i].max(), y_pred[:, i].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            
            ax.set_xlabel(f'{feature_names[i]} (True)')
            ax.set_ylabel(f'{feature_names[i]} (Predicted)')
            ax.set_title(f'{feature_names[i]}')
            ax.grid(True, alpha=0.3)
            
            # RMSE
            rmse = np.sqrt(((y_true[:, i] - y_pred[:, i]) ** 2).mean())
            ax.text(0.05, 0.95, f'RMSE: {rmse:.4f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_latent_space(z, labels=None, save_path=None):
        """Visualiza espacio latente (PCA)"""
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=2)
        z_2d = pca.fit_transform(z)
        
        plt.figure(figsize=(10, 8))
        
        if labels is not None:
            scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap='viridis', alpha=0.6, s=10)
            plt.colorbar(scatter, label='Label')
        else:
            plt.scatter(z_2d[:, 0], z_2d[:, 1], alpha=0.6, s=10)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        plt.title('Latent Space (PCA)')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        return plt.gcf()
    
    @staticmethod
    def plot_reconstruction_error(y_true, y_pred, save_path=None):
        """Grafica distribución de errores de reconstrucción"""
        error = np.abs(y_true - y_pred).mean(axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.hist(error, bins=100, alpha=0.7, edgecolor='black')
        plt.xlabel('Reconstruction Error (MAE)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Reconstruction Errors')
        plt.axvline(error.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {error.mean():.4f}')
        plt.axvline(np.percentile(error, 95), color='orange', linestyle='--', linewidth=2, label=f'95th: {np.percentile(error, 95):.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        return plt.gcf()


def save_evaluation_report(metrics, predictions=None, model_name='model'):
    """Guarda reporte de evaluación"""
    report_path = RESULT_DIR / f"{model_name}_evaluation.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"EVALUATION REPORT: {model_name}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("METRICS:\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"  {key:20s}: {value:.6f}\n")
        
        if predictions is not None:
            f.write("\nPREDICTIONS SUMMARY:\n")
            f.write(f"  Num samples: {len(predictions['y_true'])}\n")
            f.write(f"  Y shape: {predictions['y_true'].shape}\n")
    
    logger.info(f"Saved evaluation report to {report_path}")


if __name__ == '__main__':
    print("Evaluation module loaded successfully")
