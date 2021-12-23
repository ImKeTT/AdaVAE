from typing import Optional

import fire
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader

from adapters import add_bert_adapters, AdapterConfig, freeze_all_parameters, unfreeze_bert_adapters
from data import DataFrameTextClassificationDataset
from models import AutoModelForSequenceClassificationFinetuner


def train(
    num_epochs: int = 5,
    n_workers: int = 4,
    gpus: int = 1,
    precision: int = 32,
    patience: int = 5,
    adapter_size: Optional[int] = 64,
    lr: float = 2e-05,
    model_name: str = 'bert-base-multilingual-cased',
    train_file: str = './data/rusentiment/rusentiment_random_posts.csv',
    test_file: str = './data/rusentiment/rusentiment_test.csv',
    output_dir: str = './output_dir'
):
    torch.random.manual_seed(42)

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Take 10% of training data as validation
    val_df = train_df.iloc[int(len(train_df) * 0.9):]
    train_df = train_df.iloc[:int(len(train_df) * 0.9)]

    train_dataset = DataFrameTextClassificationDataset(train_df)
    val_dataset = DataFrameTextClassificationDataset(val_df)
    test_dataset = DataFrameTextClassificationDataset(test_df)

    train_loader = DataLoader(train_dataset, num_workers=n_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, num_workers=n_workers)
    test_loader = DataLoader(test_dataset, num_workers=n_workers)

    # Load pre-trained model (weights)
    model = AutoModelForSequenceClassificationFinetuner(model_name,
                                                        n_classes=train_dataset.n_classes,
                                                        lr=lr)
    if adapter_size is not None:
        # Add adapters and freeze all layers
        config = AdapterConfig(
            hidden_size=768, adapter_size=adapter_size,
            adapter_act='relu', adapter_initializer_range=1e-2
        )
        model.model.bert = add_bert_adapters(model.model.bert, config)
        model.model.bert = freeze_all_parameters(model.model.bert)

        # Unfreeze adapters and the classifier head
        model.model.bert = unfreeze_bert_adapters(model.model.bert)
        model.model.classifier.requires_grad = True
    else:
        print("Warning! BERT adapters aren't used because adapter_size wasn't specified.")

    trainer = pl.Trainer(max_epochs=num_epochs,
                         gpus=gpus,
                         auto_select_gpus=gpus > 0,
                         auto_scale_batch_size=True,
                         precision=precision,
                         callbacks=[EarlyStopping(monitor='val_loss', patience=patience)])
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    model.save_inference_artifact(output_dir)


if __name__ == '__main__':
    fire.Fire(train)
