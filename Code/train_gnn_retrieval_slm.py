import torch
from torch.utils.data import DataLoader
from component.gnn_retrieval_slm import GNNRetrievalSLM
from component.retriever import StoryRetriever
from component import utils_gnn
import os
import math

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set Working Directory
current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)

os.makedirs(os.path.join(parent_directory, "Models/GNN_Retrieval_SLM"), exist_ok=True)

data_directory = os.path.join(parent_directory, 'Data')
model_directory = os.path.join(parent_directory, 'Models/GNN_Retrieval_SLM')
print(f"Data Directory: {data_directory}")
print(f"Model Directory: {model_directory}")


def linear_warmup_cosine_lr(optimizer, num_training_steps, warmup_steps=None):
    """Create learning rate scheduler with linear warmup and cosine annealing"""
    if warmup_steps is None:
        warmup_steps = int(0.1 * num_training_steps)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, -1)


def main():
    """Main training pipeline for GNN + Retrieval + SLM model"""

    print("\n" + "=" * 60)
    print("GNN + RETRIEVAL + GPT-2 TRAINING")
    print("=" * 60)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

    # Load data
    print("\nLoading datasets...")
    train_graphs = torch.load('graph_data/train_graphs.pt', weights_only=False)
    val_graphs = torch.load('graph_data/val_graphs.pt', weights_only=False)

    print(f"  Train: {len(train_graphs):,} graphs")
    print(f"  Val: {len(val_graphs):,} graphs")

    # Create dataloaders
    batch_size = 4
    train_loader = DataLoader(
        train_graphs,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=utils_gnn.collate_fn,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_graphs,
        batch_size=batch_size,
        collate_fn=utils_gnn.collate_fn,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    # Initialize retriever
    print("\nBuilding retrieval index...")
    retriever = StoryRetriever(k=3)
    train_stories = [g.target_story for g in train_graphs]
    retriever.build_index(train_stories)

    # Initialize model (GPT-2 - no auth needed)
    print("\nInitializing GNN + Retrieval + GPT-2 model...")
    model = GNNRetrievalSLM(
        gnn_hidden_dim=256,
        slm_name='gpt2',  # ← Public model, no token needed
        retriever=retriever,
        hf_token=None  # ← Explicit None for public models
    ).to(device)

    print(f"Fusion output size: {model.fusion.out_features}")
    print(f"Prefix tokens: {model.n_prefix_tokens}")

    # Note: GPT-2 has 12 layers, not the same structure as Llama
    # Freezing logic for GPT-2:
    print("\nFreezing early GPT-2 layers...")
    if hasattr(model.slm, 'transformer') and hasattr(model.slm.transformer, 'h'):
        total_layers = len(model.slm.transformer.h)
        freeze_layers = int(0.5 * total_layers)

        for i in range(freeze_layers):
            for param in model.slm.transformer.h[i].parameters():
                param.requires_grad = False

        print(f"Frozen {freeze_layers}/{total_layers} transformer layers")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params / total_params * 100:.1f}%)")

    # Training configuration
    print("\nTraining Configuration:")
    learning_rate = 5e-5
    weight_decay = 0.01
    num_epochs = 10
    early_stopping_patience = 5

    print(f"  Learning Rate: {learning_rate}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Weight Decay: {weight_decay}")
    print(f"  Max Epochs: {num_epochs}")
    print(f"  Early Stopping Patience: {early_stopping_patience}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Learning rate scheduler
    num_training_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * num_training_steps)
    scheduler = linear_warmup_cosine_lr(optimizer, num_training_steps, warmup_steps)

    print(f"  Total Training Steps: {num_training_steps}")
    print(f"  Warmup Steps: {warmup_steps}")
    print(f"  Scheduler: Linear Warmup + Cosine Annealing (per batch)")

    # Training loop
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_epoch = 0

    os.makedirs(model_directory + '/checkpoints', exist_ok=True)

    for epoch in range(num_epochs):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'=' * 60}")

        # Train
        train_loss = utils_gnn.train_epoch(model, train_loader, optimizer, device, scheduler)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Learning Rate: {current_lr:.2e}")

        # Validate
        val_loss = utils_gnn.validate(model, val_loader, device)
        print(f"Val Loss: {val_loss:.4f}")

        # Comprehensive Evaluation
        print("\n" + "=" * 60)
        print("COMPREHENSIVE EVALUATION")
        print("=" * 60)
        try:
            results = utils_gnn.evaluate_with_metrics(model, val_loader, device, num_samples=30)
            metrics = results['metrics_summary']

            utils_gnn.display_metrics_table(metrics)
            utils_gnn.save_evaluation_results(results, model_directory, epoch)

        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            import traceback
            traceback.print_exc()

        # Save checkpoint
        checkpoint_path = f'{model_directory}/checkpoints/model_epoch_{epoch + 1}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'epochs_without_improvement': epochs_without_improvement,
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            best_model_path = f'{model_directory}/checkpoints/best_model.pt'
            torch.save(model.state_dict(), best_model_path)
            print(f"✓ New best model saved! (Val Loss: {val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s). Best Val Loss: {best_val_loss:.4f} (Epoch {best_epoch})")

        if epochs_without_improvement >= early_stopping_patience:
            print(f"\nEarly stopping triggered!")
            print(f"Best validation loss: {best_val_loss:.4f} achieved at epoch {best_epoch}")
            break

    print("\n" + "=" * 60)
    print("GNN + RETRIEVAL + GPT-2 TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()