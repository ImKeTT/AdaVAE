from pathlib import Path
from typing import List

import pytorch_lightning as pl
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification


class AutoModelForSequenceClassificationFinetuner(pl.LightningModule):
    """
    This class is used to fine-tune a pre-trained model from Transformers using PyTorch-Lightning.
    It is optimized towards models with SentencePiece tokenizers to be converted into LibTorch + SentencePiece
    for C++ deployments.

    :arg model_name: str, pre-trained model name for a model from Transformers
    :arg n_classes: int, number of classes in the classification problem
    :arg max_length: int, maximum length of tokens that the model uses
    :arg lr: float, learning rate for fine-tuning
    :arg eps: float, epsilon parameter for Adam optimizer
    """
    def __init__(self,
                 model_name: str,
                 n_classes: int,
                 max_length: int = 100,
                 lr: float = 2e-05,
                 eps: float = 1e-08):
        super(AutoModelForSequenceClassificationFinetuner, self).__init__()
        self.max_length = max_length
        self.lr = lr
        self.eps = eps

        config = AutoConfig.from_pretrained(model_name,
                                            num_labels=n_classes,
                                            output_attentions=False,
                                            output_hidden_states=False,
                                            torchscript=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    def forward(self, input_ids, attention_mask=None, labels=None):
        output = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        return output

    def tokenize(self, texts: List[str]):
        x_tokenized = self.tokenizer(texts, padding=True, truncation=True,
                                     max_length=self.max_length,
                                     return_tensors='pt')
        input_ids = x_tokenized['input_ids'].to(self.device)
        attention_mask = x_tokenized['attention_mask'].to(self.device)
        return input_ids, attention_mask

    def compute(self, batch):
        x = batch['x']
        y = batch['y']

        loss, logits = self(*self.tokenize(x), labels=y)
        return loss, logits

    def training_step(self, batch, batch_nb):
        loss, logits = self.compute(batch)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        loss, logits = self.compute(batch)

        y = batch['y']
        a, y_hat = torch.max(logits, dim=1)
        val_acc = accuracy_score(y_hat.cpu(), y.cpu())

        return {'val_loss': loss, 'val_acc': torch.tensor(val_acc)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        self.log('val_loss', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('avg_val_acc', avg_val_acc, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_nb):
        loss, logits = self.compute(batch)

        y = batch['y']
        a, y_hat = torch.max(logits, dim=1)
        test_acc = accuracy_score(y_hat.cpu(), y.cpu())

        return {'test_acc': torch.tensor(test_acc)}

    def test_epoch_end(self, outputs):
        avg_test_acc = torch.stack([x['test_acc'] for x in outputs]).mean()

        self.log('avg_test_acc', avg_test_acc, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad],
                                lr=self.lr, eps=self.eps)

    def save_inference_artifact(self, output_dir: str):
        if 'sp_model' not in dir(self.tokenizer):
            print('Warning! Model doesn\'t use SentencePiece tokenizer.')

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        # Save tokenizer
        self.tokenizer.save_vocabulary(output_dir)

        # Save TorchScript model
        self.cpu().eval()
        dummy_input_ids, _ = self.tokenize(["Simple dummy text to be used for tracing"])
        self.to_torchscript(file_path=str(output_dir / "model.pt"),
                            method='trace',
                            example_inputs=(dummy_input_ids,))
