import torch
import numpy as np
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import PLOT_DIR, RESULT_DIR

logger = logging.getLogger(__name__)


class ModelEvaluator:
    def __init__(self, model, device='cpu', is_autoencoder=False):
        self.model          = model
        self.device         = device
        self.is_autoencoder = is_autoencoder

    @torch.no_grad()
    def evaluate(self, test_loader, return_predictions=False):
        self.model.eval()
        y_true_list, y_pred_list, z_list = [], [], []

        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(self.device)
            output  = self.model(X_batch)

            if isinstance(output, tuple):
                y_pred = output[0]
                z_list.append(output[1].cpu().numpy())
            elif isinstance(output, dict):
                y_pred = output.get('pred', output.get('moe_output', output.get('shock_prob')))
                if 'latent' in output:
                    z_list.append(output['latent'].cpu().numpy())
            else:
                y_pred = output

            target = X_batch if self.is_autoencoder else Y_batch.to(self.device)
            y_true_list.append(target.cpu().numpy())
            y_pred_list.append(y_pred.cpu().numpy())

        y_true = np.concatenate(y_true_list, axis=0)
        y_pred = np.concatenate(y_pred_list, axis=0)

        n_cols = min(y_true.shape[1], y_pred.shape[1])
        y_true = y_true[:, :n_cols]
        y_pred = y_pred[:, :n_cols]

        metrics = {
            'mse':  float(mean_squared_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae':  float(mean_absolute_error(y_true, y_pred)),
        }
        names = ['Cp', 'Cfx', 'Cfy', 'Cfz'] if not self.is_autoencoder else None
        for i in range(n_cols):
            tag = names[i] if (names and i < len(names)) else f'feat_{i}'
            metrics[f'r2_{tag}'] = float(r2_score(y_true[:, i], y_pred[:, i]))

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
        for k, v in metrics.items():
            if isinstance(v, float):
                logger.info(f"  {k:20s}: {v:.6f}")
        logger.info("=" * 60)


class VisualizationTools:
    @staticmethod
    def plot_losses(train_losses, val_losses, save_path=None):
        n_train = len(train_losses)
        n_val   = len(val_losses)
        train_x = range(1, n_train + 1)
        # val is recorded every k epochs — infer k from the length ratio
        if n_val > 0 and n_val < n_train:
            step  = round(n_train / n_val)
            val_x = range(step, n_train + 1, step)
        else:
            val_x = range(1, n_val + 1)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_x, train_losses, label='Train', linewidth=2)
        ax.plot(val_x,   val_losses,   label='Val',   linewidth=2,
                marker='o', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Training and Validation Loss')
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved {save_path}")
        plt.close(fig)
        return fig

    @staticmethod
    def plot_predictions_vs_truth(y_true, y_pred, feature_names=None, save_path=None):
        if feature_names is None:
            feature_names = ['Cp', 'Cfx', 'Cfy', 'Cfz']
        n = min(y_true.shape[1], 4)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        for i in range(n):
            ax = axes[i]
            ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.3, s=1)
            lo = min(y_true[:, i].min(), y_pred[:, i].min())
            hi = max(y_true[:, i].max(), y_pred[:, i].max())
            ax.plot([lo, hi], [lo, hi], 'r--', lw=2)
            rmse = np.sqrt(((y_true[:, i] - y_pred[:, i]) ** 2).mean())
            ax.set_title(f'{feature_names[i]}  RMSE={rmse:.4f}')
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved {save_path}")
        plt.close(fig)
        return fig

    @staticmethod
    def plot_latent_space(z, X_raw=None, save_path=None):
        from sklearn.decomposition import PCA
        pca  = PCA(n_components=2)
        z_2d = pca.fit_transform(z)
        var0 = pca.explained_variance_ratio_[0]
        var1 = pca.explained_variance_ratio_[1]

    @staticmethod
    def _latent_scatter(fig, axes, z_2d, X, features, var0, var1):
        for ax, (col, label, cmap) in zip(axes, features):
            sc = ax.scatter(z_2d[:, 0], z_2d[:, 1],
                            c=X[:, col], cmap=cmap,
                            alpha=0.4, s=2, rasterized=True)
            fig.colorbar(sc, ax=ax, label=label, pad=0.02)
            ax.set_xlabel(f'PC1 ({var0:.1%})', fontsize=8)
            ax.set_ylabel(f'PC2 ({var1:.1%})', fontsize=8)
            ax.set_title(label, fontsize=10, fontweight='bold')
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2)

    @staticmethod
    def plot_latent_space(z, X_raw=None, save_path=None):
        from sklearn.decomposition import PCA
        pca  = PCA(n_components=2)
        z_2d = pca.fit_transform(z)
        var0 = pca.explained_variance_ratio_[0]
        var1 = pca.explained_variance_ratio_[1]

        RAW_FEATURES = [
            (0, 'x',       'viridis'),
            (1, 'y',       'viridis'),
            (2, 'z',       'viridis'),
            (3, 'nx',      'RdBu'),
            (4, 'ny',      'RdBu'),
            (5, 'nz',      'RdBu'),
            (6, 'Mach',    'plasma'),
            (7, 'AoA',     'coolwarm'),
            (8, 'Pi×1e-5', 'inferno'),
        ]
        DERIVED_FEATURES = [
            (9,  'q_dyn',    'plasma'),
            (10, 'Pi_norm',  'inferno'),
            (11, 'AoA_sin',  'coolwarm'),
            (12, 'L_factor', 'viridis'),
            (13, 'Cp_crit',  'RdYlBu_r'),
        ]

        has_raw     = X_raw is not None and X_raw.shape[1] >= 9
        has_derived = X_raw is not None and X_raw.shape[1] >= 14

        if has_raw:
            fig1, axes1 = plt.subplots(3, 3, figsize=(15, 13))
            VisualizationTools._latent_scatter(fig1, axes1.flatten(), z_2d, X_raw, RAW_FEATURES, var0, var1)
            fig1.suptitle('Latent Space (PCA) — original features', fontsize=13, y=1.01)
            plt.tight_layout()
            p1 = str(save_path).replace('.png', '_raw.png') if save_path else None
            if p1:
                fig1.savefig(p1, dpi=150, bbox_inches='tight')
                logger.info(f"Saved {p1}")
            plt.close(fig1)

        if has_derived:
            fig2, axes2 = plt.subplots(2, 3, figsize=(15, 9))
            axes2_flat = axes2.flatten()
            VisualizationTools._latent_scatter(fig2, axes2_flat[:5], z_2d, X_raw, DERIVED_FEATURES, var0, var1)
            axes2_flat[5].set_visible(False)
            fig2.suptitle('Latent Space (PCA) — derived physics features', fontsize=13, y=1.01)
            plt.tight_layout()
            p2 = str(save_path).replace('.png', '_derived.png') if save_path else None
            if p2:
                fig2.savefig(p2, dpi=150, bbox_inches='tight')
                logger.info(f"Saved {p2}")
            plt.close(fig2)

        if not has_raw:
            fig, ax = plt.subplots(figsize=(9, 7))
            ax.scatter(z_2d[:, 0], z_2d[:, 1], alpha=0.5, s=3, rasterized=True)
            ax.set_xlabel(f'PC1 ({var0:.1%})')
            ax.set_ylabel(f'PC2 ({var1:.1%})')
            ax.set_title('Latent Space (PCA)')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"Saved {save_path}")
            plt.close(fig)

    @staticmethod
    def plot_reconstruction_error(y_true, y_pred, save_path=None):
        error = np.abs(y_true - y_pred).mean(axis=1)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(error, bins=100, alpha=0.7, edgecolor='black')
        ax.axvline(error.mean(),               color='r',      linestyle='--', lw=2, label=f'Mean: {error.mean():.4f}')
        ax.axvline(np.percentile(error, 95),   color='orange', linestyle='--', lw=2, label=f'P95:  {np.percentile(error, 95):.4f}')
        ax.set_xlabel('MAE')
        ax.set_ylabel('Frequency')
        ax.set_title('Reconstruction Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved {save_path}")
        plt.close(fig)
        return fig


def save_evaluation_report(metrics, predictions=None, model_name='model'):
    path = RESULT_DIR / f"{model_name}_evaluation.txt"
    with open(path, 'w') as f:
        f.write(f"EVALUATION REPORT: {model_name}\n{'='*60}\n\n")
        f.write("METRICS:\n")
        for k, v in metrics.items():
            if isinstance(v, float):
                f.write(f"  {k:20s}: {v:.6f}\n")
        if predictions is not None:
            f.write(f"\nSamples: {len(predictions['y_true'])}\n")
    logger.info(f"Report saved to {path}")
