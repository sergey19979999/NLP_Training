import argparse
import os
import torch
import torch.nn as nn

from download_text import get_dataloader_and_vocab
from trainer import Trainer
from helper import (
    get_model_class,
    get_optimizer_class,
    get_lr_scheduler,
    save_vocab,
)

from download_text import ARTICLES

config = {
    "model_name": "cbow",
    "train_batch_size": 2,
    "val_batch_size": 2,
    "shuffle": True,
    "optimizer": "Adam",
    "learning_rate": 0.025,
    "epochs": 1000,
    "model_dir": "/Users/sergiomurino/Documents/NLP_Training/NLP_Training/Word2Vec_sardinian/model_folder",
    "train_steps": None,
    "val_steps": None,
    "checkpoint_frequency": None
}

def train(config):
    breakpoint()
    if not os.path.exists(config["model_dir"]):
        os.makedirs(config["model_dir"])
    
    train_dataloader, vocab = get_dataloader_and_vocab(
        articles=ARTICLES,
        model_name=config["model_name"],
        batch_size=config["train_batch_size"],
        shuffle=config["shuffle"],
        vocab=None,
    )

    val_dataloader, _ = get_dataloader_and_vocab(
        articles=ARTICLES,
        model_name=config["model_name"],
        batch_size=config["val_batch_size"],
        shuffle=config["shuffle"],
        vocab=vocab,
    )

    vocab_size = len(vocab.get_stoi())
    print(f"Vocabulary size: {vocab_size}")

    model_class = get_model_class(config["model_name"])
    model = model_class(vocab_size=vocab_size)
    criterion = nn.CrossEntropyLoss()

    optimizer_class = get_optimizer_class(config["optimizer"])
    optimizer = optimizer_class(model.parameters(), lr=config["learning_rate"])
    lr_scheduler = get_lr_scheduler(optimizer, config["epochs"], verbose=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(
        model=model,
        epochs=config["epochs"],
        train_dataloader=train_dataloader,
        train_steps=config["train_steps"],
        val_dataloader=val_dataloader,
        val_steps=config["val_steps"],
        criterion=criterion,
        optimizer=optimizer,
        checkpoint_frequency=config["checkpoint_frequency"],
        lr_scheduler=lr_scheduler,
        device=device,
        model_dir=config["model_dir"],
        model_name=config["model_name"],
    )

    trainer.train()
    print("Training finished.")

    trainer.save_model()
    trainer.save_loss()
    save_vocab(vocab, config["model_dir"])
    print("Model artifacts saved to folder:", config["model_dir"])
    
    
if __name__ == '__main__':
    train(config)