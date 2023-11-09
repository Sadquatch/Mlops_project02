
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb
import argparse
from GLUE_Transformer import GLUEDataModule, GLUETransformer


parser = argparse.ArgumentParser(description="Training script")
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
parser.add_argument('--adam_epsilon', type=float, default=1e-5, help='Adam epsilon')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay') 
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
args = parser.parse_args()

def train_model():
    # Initialize WandB for this run
    #wandb.init(project="MLops-test2")
    seed_everything(42)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir, 
        filename='{epoch}-{val_loss:.2f}-{other_metric:.2f}'
    )

    # Define your model and datamodule
    dm = GLUEDataModule(
        model_name_or_path="distilbert-base-uncased",
        task_name="mrpc",
        #train_batch_size=wandb.config.train_batch_size,
    )
    dm.setup("fit")

    model = GLUETransformer(
        model_name_or_path="distilbert-base-uncased",
        num_labels=dm.num_labels,
        eval_splits=dm.eval_splits,
        task_name=dm.task_name,
        learning_rate=wandb.config.learning_rate,
        weight_decay=wandb.config.weight_decay,
        adam_epsilon=wandb.config.adam_epsilon,
        # dropout_rate=wandb.config.dropout_rate,
    )

    # Create a Trainer with the WandbLogger
    trainer = Trainer(
        logger=WandbLogger(),
        max_epochs=3,  # Adjust this as needed
        accelerator="auto",
        gpus=1 if torch.cuda.is_available() else None,
        callbacks=[checkpoint_callback],
    )

    # Fit the model
    trainer.fit(model, datamodule=dm)

def main():
    config = dict(
            learning_rate=args.lr,
            adam_epsilon=args.adam_epsilon,
            weight_decay=args.weight_decay,
        )
    wandb.init(project='mlops_project02', config=config)
    train_model()
    #wandb.agent(sweep_id, train_model)
    wandb.finish()

if __name__ == "__main__":
    main()