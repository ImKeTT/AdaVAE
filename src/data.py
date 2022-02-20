import pandas as pd
from torch.utils.data import Dataset, DataLoader
import random
import os
import torch
import functools


class DataFrameTextClassificationDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 x_label: str = 'text',
                 y_label: str = 'label'):
        self.x = df[x_label]
        self.length = len(self.x)

        self.y = df[y_label].astype('category')
        self.n_classes = len(self.y.cat.categories)
        self.y = self.y.cat.codes

    def __getitem__(self, index) -> dict:
        x = self.x.iloc[index]
        y = self.y.iloc[index]
        return {'x': str(x), 'y': int(y)}

    def __len__(self):
        return self.length

    @staticmethod
    def from_file(file_path: str,
                  x_label: str = 'text',
                  y_label: str = 'label'):
        df = pd.read_csv(file_path)
        return DataFrameTextClassificationDataset(df, x_label, y_label)


class TextDataset_2Tokenizers_LCtrlG(Dataset):
    def __init__(self, tokenizers, args, file_path='train', text_split_mode='natural', block_size=512, create_new=0):
        """
        taken from Optimus codes
        :param tokenizers:
        :param args:
        :param file_path:
        :param text_split_mode:
        :param block_size:
        :param create_new:
        """
        print(file_path)
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory, f'cached_lm_gpt_bert_{block_size}_{filename}')

        self.examples = []
        self.tokenizers = tokenizers

        # GPT tokenizers
        self.pad_token=tokenizers[1].convert_tokens_to_ids([tokenizers[1].pad_token])[0]
        self.bos_token=tokenizers[1].convert_tokens_to_ids([tokenizers[1].bos_token])[0]
        self.eos_token=tokenizers[1].convert_tokens_to_ids([tokenizers[1].eos_token])[0]

        if not create_new and os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            if text_split_mode == 'natural':
                if args.dataset == 'Yelp':
                    dropped = self._read_corpus_natural_split_yelp(fname=file_path, label=True, max_length=block_size, block_size=block_size)
                    logger.info("The number of dropped sentences is %d", dropped)
                elif args.dataset == 'yahoo':
                    pass
                else:
                    raise NotImplementedError
            else:
                raise ValueError('Please specify the mode to split the raw text')

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, 'wb') as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        # pdb.set_trace()
        # Convert to Tensors and build dataset
        tokenized_text0= torch.tensor(self.examples[item][0], dtype=torch.long)
        tokenized_text1= torch.tensor(self.examples[item][2], dtype=torch.long)
        tokenized_text_lengths = torch.tensor([self.examples[item][1], self.examples[item][3]], dtype=torch.long)
        label = torch.tensor(self.examples[item][4], dtype=torch.long)

        # pdb.set_trace()
        return (tokenized_text0, tokenized_text1, tokenized_text_lengths, label)

    def get_labels(self):
        return ['0', '1']

    def _read_corpus_natural_split_yelp(self, fname, label, max_length, block_size):
        # label: the file contains labels.
        dropped = 0
        label_fname = fname.replace('.text', '.labels')

        with open(fname) as fin, open(label_fname) as lfin:
            for line, label_line in zip(fin, lfin):
                # pdb.set_trace()
                split_line_text = line
                lb = int(label_line)
                assert lb in [0, 1]   # binary sentiment in yelp dataset.

                if len(split_line_text) < 1:
                    dropped += 1
                    continue

                if max_length:
                    if len(split_line_text.split()) > max_length:
                        dropped += 1
                        continue

                # tokenize by tokenizers[0]
                tokenized_text0 = self.tokenizers[0].convert_tokens_to_ids(self.tokenizers[0].tokenize(split_line_text))
                tokenized_text0 = self.tokenizers[0].add_special_tokens_single_sentence(tokenized_text0)
                tokenized_text0_length = len(tokenized_text0)
                pad_token=self.tokenizers[0].convert_tokens_to_ids([self.tokenizers[0].pad_token])[0]
                # pad to max_seq_length (block_size)
                if block_size > tokenized_text0_length:
                    tokenized_text0 = tokenized_text0 + ([pad_token] * (block_size - tokenized_text0_length)  ) # Pad up to the sequence length.
                else:
                    dropped += 1
                    continue
                assert len(tokenized_text0) == block_size

                # tokenize by tokenizers[1]
                tokenized_text1 = self.tokenizers[1].convert_tokens_to_ids(self.tokenizers[1].tokenize(split_line_text))
                tokenized_text1 = self.tokenizers[1].add_special_tokens_single_sentence(tokenized_text1)
                tokenized_text1 = [self.bos_token] + tokenized_text1 + [self.eos_token]
                tokenized_text1_length = len(tokenized_text1)
                # pad to max_seq_length (block_size)
                if block_size > tokenized_text1_length:
                    tokenized_text1 = tokenized_text1 + ([self.pad_token] *  (block_size - tokenized_text1_length) ) # Pad up to the sequence length.
                else:
                    dropped += 1
                    continue
                assert len(tokenized_text1) == block_size

                self.examples.append([tokenized_text0, tokenized_text0_length, tokenized_text1, tokenized_text1_length, lb])

        return dropped

class ConditionalGenerationDataset(Dataset):
    def __init__(self, dl: list):
        self.x = []
        self.text_len = []
        self.y = []
        self.init_data(dl)
        self.length = len(self.x)

    def init_data(self, dl):
        for inst in dl:
            inst = inst.split('\t')
            ## label
            self.y.append(inst[0])
            self.x.append(inst[1])
            self.text_len.append(len(inst[1].split()))

    def __getitem__(self, index: int) -> dict:
        ## add BOS and EOS special token
        x = '<|endoftext|> ' + self.x[index][:-1] + ' <|endoftext|>'
        y = self.y[index]

        return {'x': str(x), 'y': int(y)}

    def __len__(self):
        return self.length

    ## call for direct input
    @staticmethod
    def from_file(file_path: str):
        with open(file_path, 'r') as f:
            dl = f.readlines()
        return ConditionalGenerationDataset(dl)

class GenerationDataset(Dataset):
    def __init__(self, dl: list):
        self.x = []
        self.text_len = []
        self.init_data(dl)
        self.length = len(self.x)

    def init_data(self, dl):
        for inst in dl:
            ## label
            self.x.append(inst)
            self.text_len.append(len(inst.split()))

    def __getitem__(self, index: int) -> dict:
        ## add BOS and EOS special token
        x = '<|endoftext|> ' + self.x[index] + ' <|endoftext|>'

        return {'x': str(x)}

    def __len__(self):
        return self.length

    ## call for direct input
    @staticmethod
    def from_file(file_path: str):
        with open(file_path, 'r') as f:
            dl = f.readlines()
        return GenerationDataset(dl)

class DialogGenerationDataset(Dataset):
    def __init__(self, dl: list):
        self.x = []
        self.text_len = []
        self.y = []
        self.init_data(dl)
        self.length = len(self.x)

    def init_data(self, dl):
        for inst in dl:
            inst = inst.split('\t')
            ## context
            self.y.append(inst[0])
            ## response
            self.x.append(inst[1])
            self.text_len.append(len(inst[1].split()))

    def __getitem__(self, index: int) -> dict:
        ## add BOS and EOS special token
        x = '<|endoftext|> ' + self.x[index] + ' <|endoftext|>'
        y = '<|endoftext|> ' + self.y[index] + ' <|endoftext|>'

        return {'response': str(x), 'context': int(y)}

    def __len__(self):
        return self.length

    ## call for direct input
    @staticmethod
    def from_file(file_path: str):
        with open(file_path, 'r') as f:
            dl = f.readlines()
        return DialogGenerationDataset(dl)

def collate_fn(samples: dict, eos_id: list, tokenizer):
    """ Creates a batch out of samples for direct input"""
    x_max_len = max(map(lambda s: len(s['x']), samples))
    # Zero pad mask
    x_mask = torch.ByteTensor([[1] * len(ss['x']) + [0] * (x_max_len - len(ss['x'])) for ss in samples])
    # tokenizer.convert_tokens_to_ids('<|startoftext|>') = 50257, endoftext 50256, use 50257 here causes errors!!
    x = torch.LongTensor([ss['x'] + eos_id * (x_max_len - len(ss['x'])) for ss in samples])


def prepare_dataset(data_dir, dataset_name, tokenizer, train_bsz, train_seq_len, val_bsz, val_seq_len, test_bsz=1,
                    test_seq_len=1024, data_type='t0', num_workers=1, make_train=True, make_val=True, make_test=False):
    loaders = []
    if make_train:
        train_dataset = ConditionalGenerationDataset.from_file('./data/yelp_polarity/train.txt')