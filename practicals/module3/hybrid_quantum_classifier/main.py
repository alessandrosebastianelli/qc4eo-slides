"""
Main script for Hybrid Quantum Classifier on EuroSAT Dataset

This script orchestrates the entire pipeline:
1. Data loading and preprocessing
2. Quantum device setup
3. Model creation
4. Training
5. Testing and evaluation
6. Visualization
"""

import torch
import numpy as np
import pytorch_lightning as pl

import sys

sys.path += ['.', '..']

# Import project modules
from config import Config
from data import download_eurosat, load_images, create_dataloaders
from quantum import create_quantum_device
from models import HybridQuantumClassifier
from training import train_model, test_model, save_model, print_training_summary
from utils import plot_sample_images, visualize_results


def set_random_seeds(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    pl.seed_everything(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    """
    Main execution function.
    
    This function runs the complete pipeline from data loading to evaluation.
    """
    print("\n" + "=" * 60)
    print("HYBRID QUANTUM CLASSIFIER FOR SATELLITE IMAGERY")
    print("EuroSAT Dataset - 10 Land Use Classes")
    print("=" * 60 + "\n")
    
    # ==================== CONFIGURATION ====================
    
    print("Step 1: Loading Configuration...")
    config = Config()
    config.print_config()
    
    # Set random seeds for reproducibility
    set_random_seeds(42)
    print("✓ Random seeds set for reproducibility\n")
    
    # Check for GPU
    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}\n")
    else:
        print("⚠ No GPU available, using CPU\n")
    
    # ==================== DATA LOADING ====================
    
    print("\n" + "=" * 60)
    print("Step 2: Loading Data")
    print("=" * 60 + "\n")
    
    # Download dataset if needed
    download_eurosat()
    
    # Load images
    images, labels_str, class_names = load_images(
        max_per_class=config.MAX_IMAGES_PER_CLASS
    )
    
    # Update config with actual number of classes
    config.N_CLASSES = len(class_names)
    
    # Visualize sample images
    print("\nVisualizing sample images...")
    from data.loader import create_label_mapping, labels_to_numeric
    label_to_idx = create_label_mapping(class_names)
    labels_numeric = labels_to_numeric(labels_str, label_to_idx)
    plot_sample_images(images, labels_numeric, class_names, n_samples=10)
    
    # Create data loaders
    train_loader, val_loader, test_loader, label_to_idx = create_dataloaders(
        images, labels_str, class_names, config
    )
    
    print(f"\n✓ Data loading complete!")
    print(f"  Classes: {class_names}")
    print(f"  Total samples: {len(images)}")
    
    # ==================== QUANTUM DEVICE ====================
    
    print("\n" + "=" * 60)
    print("Step 3: Setting up Quantum Device")
    print("=" * 60)
    
    quantum_device = create_quantum_device(
        n_qubits=config.N_QUBITS,
        device_name=config.DEVICE
    )
    
    # ==================== MODEL CREATION ====================
    
    print("\n" + "=" * 60)
    print("Step 4: Creating Hybrid Model")
    print("=" * 60 + "\n")
    
    model = HybridQuantumClassifier(config, quantum_device)
    model.print_model_info()
    
    # ==================== TRAINING ====================
    
    print("\n" + "=" * 60)
    print("Step 5: Training")
    print("=" * 60)
    
    model, trainer, training_time = train_model(
        model, train_loader, val_loader, config
    )
    
    # ==================== TESTING ====================
    
    print("\n" + "=" * 60)
    print("Step 6: Testing")
    print("=" * 60)
    
    test_results = test_model(model, trainer, test_loader, class_names)
    
    # ==================== VISUALIZATION ====================
    
    print("\n" + "=" * 60)
    print("Step 7: Generating Visualizations")
    print("=" * 60)
    
    visualize_results(test_results)
    
    # ==================== SAVE MODEL ====================
    
    print("\n" + "=" * 60)
    print("Step 8: Saving Model")
    print("=" * 60)
    
    model_path = save_model(
        model=model,
        config=config,
        class_names=class_names,
        test_accuracy=test_results['accuracy'],
        training_time=training_time
    )
    
    # ==================== FINAL SUMMARY ====================
    
    print_training_summary(model, test_results, training_time, config)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\n✓ Model saved: {model_path}")
    print(f"✓ Test Accuracy: {test_results['accuracy']:.4f} ({test_results['accuracy']*100:.2f}%)")
    print(f"✓ Training Time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print("\nThank you for using the Hybrid Quantum Classifier!")
    print("=" * 60 + "\n")


def quick_test():
    """
    Quick test function to verify all components work.
    Run this to test the installation without full training.
    """
    print("\n" + "=" * 60)
    print("QUICK TEST MODE")
    print("=" * 60 + "\n")
    
    # Test imports
    print("Testing imports...")
    from config import Config
    from quantum import create_quantum_device, print_encoding_options, print_circuit_options
    print("✓ All imports successful\n")
    
    # Test configuration
    print("Testing configuration...")
    config = Config()
    config.print_config()
    
    # Test quantum device
    print("Testing quantum device...")
    device = create_quantum_device(4, 'default.qubit')
    print("✓ Quantum device created\n")
    
    # Show available options
    print_encoding_options()
    print_circuit_options()
    
    print("\n" + "=" * 60)
    print("QUICK TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60 + "\n")
    print("You can now run the full training with: python main.py")


if __name__ == "__main__":
    # You can uncomment the line below to run a quick test first
    #quick_test()
    
    # Run the full pipeline
    main()
