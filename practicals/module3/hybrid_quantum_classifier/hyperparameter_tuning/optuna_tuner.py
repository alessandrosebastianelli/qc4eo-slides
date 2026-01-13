"""
Optuna-based hyperparameter tuning for Hybrid Quantum Classifier

This module enables automatic tuning of both classical and quantum hyperparameters
using Optuna's trial-and-error optimization framework.
"""

import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate
)
import torch
import pytorch_lightning as pl

from copy import deepcopy
import json
from pathlib import Path

from config import Config
from data import download_eurosat, load_images, create_dataloaders
from quantum import create_quantum_device
from models import HybridQuantumClassifier

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping


class OptunaHybridTuner:
    """
    Optuna-based hyperparameter tuner for hybrid quantum-classical models.
    
    This class handles the optimization of both classical hyperparameters
    (learning rate, batch size, dropout, etc.) and quantum hyperparameters
    (number of qubits, layers, encoding type, circuit type).
    
    Args:
        base_config: Base configuration object to start from
        tune_quantum: Whether to tune quantum hyperparameters
        tune_classical: Whether to tune classical hyperparameters
        n_trials: Number of optimization trials
        timeout: Maximum time for optimization (seconds)
    """
    
    def __init__(
        self,
        base_config=None,
        tune_quantum=True,
        tune_classical=True,
        n_trials=50,
        timeout=None
    ):
        self.base_config = base_config or Config()
        self.tune_quantum = tune_quantum
        self.tune_classical = tune_classical
        self.n_trials = n_trials
        self.timeout = timeout
        
        # Load data once (reused across trials)
        print("Loading dataset for hyperparameter tuning...")
        download_eurosat()
        self.images, self.labels, self.class_names = load_images(
            max_per_class=self.base_config.MAX_IMAGES_PER_CLASS
        )
        
        # Best configuration found
        self.best_config = None
        self.best_value = None
    
    def objective(self, trial):
        """
        Objective function for Optuna to optimize.
        
        This function is called for each trial. It:
        1. Suggests hyperparameters
        2. Creates a model with those hyperparameters
        3. Trains the model
        4. Returns validation accuracy
        
        Args:
            trial: Optuna trial object
            
        Returns:
            float: Validation accuracy to maximize
        """
        # Create a copy of base config
        config = deepcopy(self.base_config)
        
        # ==================== QUANTUM HYPERPARAMETERS ====================
        
        if self.tune_quantum:
            # Number of qubits (4, 6, 8, 10)
            config.N_QUBITS = trial.suggest_int('n_qubits', 4, 10, step=2)
            
            # Number of layers (2, 3, 4, 5)
            config.N_LAYERS = trial.suggest_int('n_layers', 2, 5)
            
            # Encoding strategy
            config.ENCODING = trial.suggest_categorical(
                'encoding',
                ['angle', 'amplitude', 'iqp', 'basis']
            )
            
            # Circuit type
            config.CIRCUIT_TYPE = trial.suggest_categorical(
                'circuit_type',
                ['strongly_entangling', 'basic_entangling', 
                 'hardware_efficient', 'custom_alternating']
            )
        
        # ==================== CLASSICAL HYPERPARAMETERS ====================
        
        if self.tune_classical:
            # Learning rate (log scale)
            config.LEARNING_RATE = trial.suggest_float(
                'learning_rate',
                1e-5, 1e-2,
                log=True
            )
            
            # Batch size (power of 2)
            config.BATCH_SIZE = trial.suggest_categorical(
                'batch_size',
                [8, 16, 32, 64]
            )
            
            # Dropout rate
            config.DROPOUT_RATE = trial.suggest_float(
                'dropout_rate',
                0.1, 0.5
            )
            
            # CNN output dimension
            config.CNN_OUTPUT_DIM = trial.suggest_categorical(
                'cnn_output_dim',
                [128, 256, 512]
            )
        
        # ==================== TRAINING ====================
        
        # Reduce epochs for faster tuning
        config.MAX_EPOCHS = 20  # Quick training for each trial
        config.PATIENCE = 5     # Early stopping
        
        try:
            # Create dataloaders
            train_loader, val_loader, _, _ = create_dataloaders(
                self.images, self.labels, self.class_names, config
            )
            
            # Create quantum device
            quantum_device = create_quantum_device(
                n_qubits=config.N_QUBITS,
                device_name=config.DEVICE
            )
            
            # Create model
            model = HybridQuantumClassifier(config, quantum_device)
            
            # Setup trainer with minimal logging
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=config.PATIENCE,
                mode='min',
                verbose=False
            )
            
            trainer = Trainer(
                max_epochs=config.MAX_EPOCHS,
                callbacks=[early_stopping],
                accelerator='auto',
                devices=1,
                enable_progress_bar=False,  # Disable for cleaner output
                enable_model_summary=False,
                logger=False,  # Disable logging
                deterministic=True
            )
            
            # Train
            trainer.fit(model, train_loader, val_loader)
            
            # Get validation accuracy
            val_acc = trainer.callback_metrics.get('val_acc', 0.0)
            
            # Report intermediate value for pruning
            trial.report(float(val_acc), step=trainer.current_epoch)
            
            # Handle pruning
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return float(val_acc)
            
        except Exception as e:
            print(f"Trial failed with error: {e}")
            return 0.0  # Return worst possible score
    
    def tune(self, direction='maximize', study_name='hybrid_quantum_tuning'):
        """
        Run the hyperparameter optimization study.
        
        Args:
            direction: 'maximize' for accuracy, 'minimize' for loss
            study_name: Name for the Optuna study
            
        Returns:
            optuna.Study: Completed study object
        """
        print("\n" + "=" * 60)
        print("STARTING HYPERPARAMETER OPTIMIZATION")
        print("=" * 60)
        print(f"Number of trials: {self.n_trials}")
        print(f"Tuning quantum params: {self.tune_quantum}")
        print(f"Tuning classical params: {self.tune_classical}")
        print("=" * 60 + "\n")
        
        # Create study
        study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            pruner=optuna.pruners.MedianPruner(  # Prune unpromising trials
                n_startup_trials=5,
                n_warmup_steps=5
            )
        )
        
        # Optimize
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        # Store best results
        self.best_value = study.best_value
        self.best_config = self._create_best_config(study.best_params)
        
        # Print results
        self._print_results(study)
        
        return study
    
    def _create_best_config(self, best_params):
        """Create a Config object from best parameters"""
        config = deepcopy(self.base_config)
        
        # Update with best parameters
        for key, value in best_params.items():
            if key == 'n_qubits':
                config.N_QUBITS = value
            elif key == 'n_layers':
                config.N_LAYERS = value
            elif key == 'encoding':
                config.ENCODING = value
            elif key == 'circuit_type':
                config.CIRCUIT_TYPE = value
            elif key == 'learning_rate':
                config.LEARNING_RATE = value
            elif key == 'batch_size':
                config.BATCH_SIZE = value
            elif key == 'dropout_rate':
                config.DROPOUT_RATE = value
            elif key == 'cnn_output_dim':
                config.CNN_OUTPUT_DIM = value
        
        return config
    
    def _print_results(self, study):
        """Print optimization results"""
        print("\n" + "=" * 60)
        print("HYPERPARAMETER OPTIMIZATION COMPLETED")
        print("=" * 60)
        print(f"Best validation accuracy: {study.best_value:.4f}")
        print(f"Best trial: #{study.best_trial.number}")
        print("\nBest hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key:20s}: {value}")
        print("=" * 60 + "\n")
    
    def save_results(self, study, filename='tuning_results.json'):
        """
        Save tuning results to file.
        
        Args:
            study: Optuna study object
            filename: Output filename
        """
        results = {
            'best_value': study.best_value,
            'best_params': study.best_params,
            'best_trial_number': study.best_trial.number,
            'n_trials': len(study.trials),
            'all_trials': [
                {
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'state': str(trial.state)
                }
                for trial in study.trials
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {filename}")


def create_tuning_config(
    tune_quantum=True,
    tune_classical=True,
    quick_mode=False
):
    """
    Create a tuning configuration.
    
    Args:
        tune_quantum: Whether to tune quantum hyperparameters
        tune_classical: Whether to tune classical hyperparameters
        quick_mode: If True, use fewer trials for quick testing
        
    Returns:
        dict: Configuration for tuning
    """
    n_trials = 10 if quick_mode else 50
    
    return {
        'tune_quantum': tune_quantum,
        'tune_classical': tune_classical,
        'n_trials': n_trials,
        'timeout': 3600 if quick_mode else None  # 1 hour limit for quick mode
    }


def run_study(
    tune_quantum=True,
    tune_classical=True,
    n_trials=50,
    visualize=True,
    save_results=True
):
    """
    Convenience function to run a complete tuning study.
    
    Args:
        tune_quantum: Whether to tune quantum hyperparameters
        tune_classical: Whether to tune classical hyperparameters
        n_trials: Number of trials to run
        visualize: Whether to generate visualization plots
        save_results: Whether to save results to file
        
    Returns:
        tuple: (best_config, study)
    """
    # Create tuner
    tuner = OptunaHybridTuner(
        tune_quantum=tune_quantum,
        tune_classical=tune_classical,
        n_trials=n_trials
    )
    
    # Run optimization
    study = tuner.tune()
    
    # Save results
    if save_results:
        tuner.save_results(study)
    
    # Visualize
    if visualize:
        visualize_study_results(study)
    
    return tuner.best_config, study


def visualize_study_results(study, save_dir='tuning_plots'):
    """
    Create visualization plots for the tuning study.
    
    Args:
        study: Optuna study object
        save_dir: Directory to save plots
    """
    import matplotlib.pyplot as plt
    
    Path(save_dir).mkdir(exist_ok=True)
    
    print("\nGenerating visualization plots...")
    
    # 1. Optimization history
    fig1 = plot_optimization_history(study)
    fig1.write_html(f'{save_dir}/optimization_history.html')
    print(f"  ✓ Optimization history saved")
    
    # 2. Parameter importances
    try:
        fig2 = plot_param_importances(study)
        fig2.write_html(f'{save_dir}/param_importances.html')
        print(f"  ✓ Parameter importances saved")
    except:
        print(f"  ⚠ Could not generate parameter importances (needs >= 2 trials)")
    
    # 3. Parallel coordinate plot
    try:
        fig3 = plot_parallel_coordinate(study)
        fig3.write_html(f'{save_dir}/parallel_coordinate.html')
        print(f"  ✓ Parallel coordinate plot saved")
    except:
        print(f"  ⚠ Could not generate parallel coordinate plot")
    
    print(f"\nAll plots saved to: {save_dir}/")
    print(f"Open the HTML files in a browser to view interactive plots.")


def compare_configurations(configs, names=None):
    """
    Compare multiple configurations side by side.
    
    Args:
        configs: List of Config objects
        names: Optional list of names for each config
        
    Returns:
        pandas.DataFrame: Comparison table
    """
    import pandas as pd
    
    if names is None:
        names = [f"Config {i+1}" for i in range(len(configs))]
    
    data = []
    for config in configs:
        data.append({
            'N_QUBITS': config.N_QUBITS,
            'N_LAYERS': config.N_LAYERS,
            'ENCODING': config.ENCODING,
            'CIRCUIT_TYPE': config.CIRCUIT_TYPE,
            'LEARNING_RATE': config.LEARNING_RATE,
            'BATCH_SIZE': config.BATCH_SIZE,
            'DROPOUT_RATE': config.DROPOUT_RATE,
            'CNN_OUTPUT_DIM': config.CNN_OUTPUT_DIM
        })
    
    df = pd.DataFrame(data, index=names)
    return df


# Example usage patterns
if __name__ == "__main__":
    print("Optuna Hyperparameter Tuning Module")
    print("=" * 60)
    print("\nExample usage:")
    print("""
    from hyperparameter_tuning import OptunaHybridTuner, run_study
    
    # Quick example - tune everything
    best_config, study = run_study(
        tune_quantum=True,
        tune_classical=True,
        n_trials=20
    )
    
    # Use the best configuration
    print(best_config.get_config_dict())
    """)
