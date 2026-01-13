"""
Training and testing utilities using PyTorch Lightning
"""

import time
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


def train_model(model, train_loader, val_loader, config):
    """
    Train the hybrid quantum classifier.
    
    Uses PyTorch Lightning Trainer for automatic training loop,
    GPU support, logging, and callbacks.
    
    Args:
        model: HybridQuantumClassifier instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        config: Configuration object
        
    Returns:
        tuple: (trained_model, trainer, training_time)
    """
    print("\n" + "=" * 60)
    print("TRAINING STARTED")
    print("=" * 60)
    print(f"Encoding: {config.ENCODING}")
    print(f"Circuit: {config.CIRCUIT_TYPE}")
    print(f"Device: {config.DEVICE}")
    print(f"Qubits: {config.N_QUBITS}")
    print(f"Layers: {config.N_LAYERS}")
    print(f"Max Epochs: {config.MAX_EPOCHS}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print("=" * 60 + "\n")
    
    # Setup callbacks
    
    # Early Stopping: stop training if validation loss doesn't improve
    early_stopping = EarlyStopping(
        monitor='val_loss',           # Metric to monitor
        patience=config.PATIENCE,      # Number of epochs to wait
        mode='min',                    # Stop when monitored metric stops decreasing
        verbose=True
    )
    
    # Model Checkpoint: save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',                    # Metric to monitor
        mode='max',                           # Save when metric is maximum
        save_top_k=1,                         # Keep only the best model
        filename='best-model-{epoch:02d}-{val_acc:.4f}',
        verbose=True
    )
    
    # Create PyTorch Lightning Trainer
    trainer = Trainer(
        max_epochs=config.MAX_EPOCHS,
        callbacks=[early_stopping, checkpoint_callback],
        accelerator='cpu',              # Automatically use GPU if available
        devices=1,                       # Use 1 device
        log_every_n_steps=10,           # Log metrics every 10 steps
        enable_progress_bar=True,       # Show progress bar
        deterministic=True,             # For reproducibility
    )
    
    # Print model architecture
    model.print_model_info()
    
    # Start training
    print("Starting training...")
    start_time = time.time()
    
    try:
        trainer.fit(model, train_loader, val_loader)
        training_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)
        print(f"Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        print(f"Best model saved at: {checkpoint_callback.best_model_path}")
        print(f"Best validation accuracy: {checkpoint_callback.best_model_score:.4f}")
        print("=" * 60 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        training_time = time.time() - start_time
        print(f"Training time before interruption: {training_time:.2f} seconds")
    
    return model, trainer, training_time


def test_model(model, trainer, test_loader, class_names):
    """
    Test the trained model on the test set.
    
    Args:
        model: Trained HybridQuantumClassifier
        trainer: PyTorch Lightning Trainer instance
        test_loader: DataLoader for test data
        class_names: List of class names
        
    Returns:
        dict: Test results including accuracy and predictions
    """
    print("\n" + "=" * 60)
    print("TESTING MODEL")
    print("=" * 60)
    
    # Run testing
    test_results = trainer.test(model, test_loader)
    
    # Get predictions and targets (stored in model during test)
    predictions = model.test_predictions
    targets = model.test_targets
    
    # Calculate accuracy
    test_acc = (predictions == targets).mean()
    
    print(f"\nTest Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print("=" * 60 + "\n")
    
    return {
        'test_results': test_results,
        'predictions': predictions,
        'targets': targets,
        'accuracy': test_acc,
        'class_names': class_names
    }


def save_model(model, config, class_names, test_accuracy, training_time, filename=None):
    """
    Save the trained model and metadata.
    
    Args:
        model: Trained model
        config: Configuration object
        class_names: List of class names
        test_accuracy: Test accuracy achieved
        training_time: Time taken to train
        filename: Optional custom filename
        
    Returns:
        str: Path to saved model
    """
    if filename is None:
        filename = f"hybrid_quantum_{config.ENCODING}_{config.CIRCUIT_TYPE}.pth"
    
    # Create save dictionary
    save_dict = {
        'model_state_dict': model.state_dict(),
        'config': config.get_config_dict(),
        'class_names': class_names,
        'test_accuracy': test_accuracy,
        'training_time': training_time,
    }
    
    # Save
    torch.save(save_dict, filename)
    print(f"\nModel saved as: {filename}")
    
    return filename


def load_model(filename, model_class, quantum_device):
    """
    Load a saved model.
    
    Args:
        filename: Path to saved model file
        model_class: HybridQuantumClassifier class
        quantum_device: PennyLane quantum device
        
    Returns:
        tuple: (model, metadata)
    """
    # Load checkpoint
    checkpoint = torch.load(filename)
    
    # Recreate config (you'd need to create a Config object from the dict)
    # For simplicity, we'll just return the checkpoint
    print(f"Model loaded from: {filename}")
    print(f"Test accuracy: {checkpoint['test_accuracy']:.4f}")
    print(f"Training time: {checkpoint['training_time']:.2f} seconds")
    
    return checkpoint


def print_training_summary(model, test_results, training_time, config):
    """
    Print a comprehensive summary of training results.
    
    Args:
        model: Trained model
        test_results: Results from testing
        training_time: Time taken to train
        config: Configuration object
    """
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    
    print("\nConfiguration:")
    print(f"  Encoding: {config.ENCODING}")
    print(f"  Circuit: {config.CIRCUIT_TYPE}")
    print(f"  Qubits: {config.N_QUBITS}")
    print(f"  Layers: {config.N_LAYERS}")
    print(f"  Batch Size: {config.BATCH_SIZE}")
    print(f"  Learning Rate: {config.LEARNING_RATE}")
    
    print("\nModel Parameters:")
    params = model.get_num_parameters()
    for component, count in params.items():
        print(f"  {component.capitalize():15s}: {count:,}")
    
    print("\nPerformance:")
    print(f"  Test Accuracy: {test_results['accuracy']:.4f} ({test_results['accuracy']*100:.2f}%)")
    print(f"  Training Time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    print("=" * 60 + "\n")