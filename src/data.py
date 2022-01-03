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
        x = '<|endoftext|> ' + self.x[index] + ' <|endoftext|>'
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


def collate_fn_old(samples, eos_id):
    """ Creates a batch out of samples for direct input"""
    x_max_len = max(map(lambda s: len(s[0]), samples))
    # Zero pad mask
    x_mask = torch.ByteTensor([[1] * len(ss[0]) + [0] * (x_max_len - len(ss[0])) for ss in samples])
    # tokenizer.convert_tokens_to_ids('<|startoftext|>') = 50257, endoftext 50256, use 50257 here causes errors!!
    x = torch.LongTensor([ss[0] + [50256] * (x_max_len - len(ss[0])) for ss in samples])

    max_len = max(map(lambda s: len(s[1]), samples))
    # Zero pad mask
    y_mask = torch.ByteTensor([[1] * len(ss[1]) + [0] * (max_len - len(ss[1])) for ss in samples])
    # tokenizer.convert_tokens_to_ids('<|startoftext|>') = 50257
    y = torch.LongTensor([ss[1] + [50256] * (max_len - len(ss[1])) for ss in samples])

    max_len = max(map(lambda s: len(s[2]), samples))
    # Zero pad mask
    input_mask = torch.ByteTensor([[1] * len(ip[2]) + [0] * (max_len - len(ip[2])) for ip in samples])
    # tokenizer.convert_tokens_to_ids('<|startoftext|>') = 50257
    input = torch.LongTensor([ip[2] + [50256] * (max_len - len(ip[2])) for ip in samples])

    return x_mask, x, y_mask, y, input[:, :-1], input[:, 1:].contiguous(), input_mask[:, 1:]

class Preprocessor_base():
    def __init__(self):
        self.fn = None

    def make_fn(self):
        raise NotImplementedError()

    def __call__(self, x):
        try:
            if self.fn is None:
                self.fn = self.make_fn()
            x = self.fn(x)
            return x
        except Exception as e:
            print('Error in preprocessing', repr(e))
            raise e

class Preprocessor(Preprocessor_base):
    def __init__(self, tokenizer):
        pass
    def make_fn(self):
        pass


def prefix_truncate(window):
    """ truncates text to the prefix window size """

    def f(text):
        if len(text) > window:
            text = text[:window]
        return text

    return f

def truncate_tuple(truncator, t):
    return truncator(t[0]), truncator(t[1]), truncator(t[2])

def encode_tuple(tokenizer, t):
    return tokenizer.encode(t[0]), tokenizer.encode(t[1]), tokenizer.encode(t[2])

def compose(*functions):
    """ Executes a list of functions in order, using functools.reduce function """
    return functools.reduce(lambda f, g: lambda x: g(f(x)), functions, lambda x: x)

class Preprocessor1(Preprocessor_base):
    def __init__(self, tokenizer, seq_len, data_type):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.data_type = data_type

    def make_fn(self):
        """ Executes a list of functions in order, using functools.reduce function """
        return compose(
            insert_keywords(self.tokenizer, self.data_type),
            lambda input: encode_tuple(self.tokenizer, input) if isinstance(input, tuple) else [encode_tuple(self.tokenizer, inp) for inp in input],
            lambda input: truncate_tuple(prefix_truncate(self.seq_len), input) if isinstance(input, tuple) else [truncate_tuple(prefix_truncate(self.seq_len), inp) for inp in input]
        )

def prepare_dataset_old(data_dir, dataset_name, tokenizer, train_bsz, train_seq_len, val_bsz, val_seq_len, test_bsz=1,
                    test_seq_len=1024, data_type='t0', num_workers=1, make_train=True, make_val=True, make_test=False):
    # data_dir, dataset_name, tokenizer, train_bsz, train_seq_len, val_bsz, val_seq_len, num_workers = args.data_dir, args.dataset, tokenizer, batch_schedule[cur_b_schedule][0], batch_schedule[cur_b_schedule][1], batch_schedule[-1][0], batch_schedule[-1][1], args.workers

    loaders = []
    if dataset_name == 'wp':
        train_collate_fn = collate_fn
        val_collate_fn = collate_fn
        test_collate_fn = collate_fn

        if make_train:
            train_preproc = Preprocessor(tokenizer, train_seq_len, data_type)
            d_train = PromptDataset(
                os.path.join(data_dir, 'writingPrompts/train.wp_source'),
                os.path.join(data_dir, 'writingPrompts/train.wp_target'),
                train_preproc)
            if data_type == 't7' or data_type == 't8':
                d_train = [t for lt in d_train for t in lt]
            print('Train dataset size', len(d_train))
            loaders.append(DataLoader(d_train,
                                           # sampler=DistributedSampler(d_train) if distributed else None,
                                           batch_size=train_bsz,
                                           pin_memory=True,
                                           drop_last=True,
                                           num_workers=num_workers,
                                           collate_fn=train_collate_fn) if d_train else None)
        if make_val:
            val_preproc = Preprocessor(tokenizer, val_seq_len, data_type)
            d_val = PromptDataset(
                os.path.join(data_dir, 'writingPrompts/valid.wp_source'),
                os.path.join(data_dir, 'writingPrompts/valid.wp_target'),
                val_preproc)
            if data_type == 't7' or data_type == 't8':
                d_val = [t for lt in d_val for t in lt]
            print('Val dataset size', len(d_val))
            loaders.append(DataLoader(d_val,
                                           # sampler=DistributedSampler(d_val),
                                           batch_size=val_bsz,
                                           pin_memory=True,
                                           drop_last=True,
                                           num_workers=num_workers,
                                           collate_fn=val_collate_fn) if d_val else None)
        if make_test:
            test_preproc = Preprocessor(tokenizer, test_seq_len, data_type)
            d_test = PromptDataset(
                os.path.join(data_dir, 'writingPrompts/test.wp_source'),
                os.path.join(data_dir, 'writingPrompts/test.wp_target'),
                test_preproc)
            if data_type == 't7' or data_type == 't8':
                d_test = [t for lt in d_test for t in lt]
            print('Test dataset size', len(d_test))
            loaders.append(DataLoader(d_test,
                                           # sampler=DistributedSampler(d_val),
                                           batch_size=test_bsz,
                                           pin_memory=True,
                                           drop_last=True,
                                           num_workers=num_workers,
                                           collate_fn=test_collate_fn) if d_test else None)
    elif dataset_name == 'wi':
        train_collate_fn = collate_fn
        val_collate_fn = collate_fn
        test_collate_fn = collate_fn

        print('Loading wikiplot dataset...')
        data_plots = os.path.join(data_dir, 'wikiPlots/plots_paragraph')
        data_titles = os.path.join(data_dir, 'wikiPlots/titles')
        with open(data_plots, errors='ignore') as fp:
            plots = fp.readlines()
        with open(data_titles, errors='ignore') as ft:
            titles = ft.readlines()

        texts = [(t, p) for t, p in zip(titles, plots) if t.strip() != '' and p.strip() != '']
        print('Done.')
        train_text = texts[:int(len(texts) * 0.9)]
        val_text = texts[int(len(texts) * 0.9):int(len(texts) * 0.95)]
        test_text = texts[int(len(texts) * 0.95):]

        if make_train:
            train_preproc = Preprocessor(tokenizer, train_seq_len, data_type)
            d_train = PlotDataset(train_text, train_preproc)
            if data_type == 't7' or data_type == 't8':
                d_train = [t for lt in d_train for t in lt]
            print('Train dataset size', len(d_train))
            loaders.append(DataLoader(d_train,
                                           # sampler=DistributedSampler(d_train) if distributed else None,
                                           batch_size=train_bsz,
                                           pin_memory=True,
                                           drop_last=True,
                                           num_workers=num_workers,
                                           collate_fn=train_collate_fn) if d_train else None)
        if make_val:
            val_preproc = Preprocessor(tokenizer, val_seq_len, data_type)
            d_val = PlotDataset(val_text, val_preproc)
            if data_type == 't7' or data_type == 't8':
                d_val = [t for lt in d_val for t in lt]
            print('Val dataset size', len(d_val))
            loaders.append(DataLoader(d_val,
                                           # sampler=DistributedSampler(d_val),
                                           batch_size=val_bsz,
                                           pin_memory=True,
                                           drop_last=True,
                                           num_workers=num_workers,
                                           collate_fn=val_collate_fn) if d_val else None)
        if make_test:
            test_preproc = Preprocessor(tokenizer, test_seq_len, data_type)
            d_test = PlotDataset(test_text, test_preproc)
            if data_type == 't7' or data_type == 't8':
                d_test = [t for lt in d_test for t in lt]
            print('Test dataset size', len(d_test))
            loaders.append(DataLoader(d_test,
                                           # sampler=DistributedSampler(d_val),
                                           batch_size=test_bsz,
                                           pin_memory=True,
                                           drop_last=True,
                                           num_workers=num_workers,
                                           collate_fn=test_collate_fn) if d_test else None)
    elif dataset_name == 'ax':
        train_collate_fn = collate_fn
        val_collate_fn = collate_fn
        test_collate_fn = collate_fn

        print('Loading arxiv dataset...')
        data_abs = os.path.join(data_dir, 'arxiv/artificial intelligence_10047_15000_15_abs.txt')
        data_titles = os.path.join(data_dir, 'arxiv/artificial intelligence_10047_15000_15_title.txt')
        with open(data_abs, errors='ignore') as fp:
            abs = fp.readlines()
        with open(data_titles, errors='ignore') as ft:
            titles = ft.readlines()
        assert len(titles) == len(abs)
        ai_data = [('ai', t.strip(), p.strip()) for t, p in zip(titles, abs) if t.strip() != '' and p.strip() != '']

        data_abs = os.path.join(data_dir, 'arxiv/computer vision_14582_15000_15_abs.txt')
        data_titles = os.path.join(data_dir, 'arxiv/computer vision_14582_15000_15_title.txt')
        with open(data_abs, errors='ignore') as fp:
            abs = fp.readlines()
        with open(data_titles, errors='ignore') as ft:
            titles = ft.readlines()
        assert len(titles) == len(abs)
        cv_data = [('cv', t.strip(), p.strip()) for t, p in zip(titles, abs) if t.strip() != '' and p.strip() != '']

        data_abs = os.path.join(data_dir, 'arxiv/language generation_14514_15000_15_abs.txt')
        data_titles = os.path.join(data_dir, 'arxiv/language generation_14514_15000_15_title.txt')
        with open(data_abs, errors='ignore') as fp:
            abs = fp.readlines()
        with open(data_titles, errors='ignore') as ft:
            titles = ft.readlines()
        assert len(titles) == len(abs)
        lg_data = [('lg', t.strip(), p.strip()) for t, p in zip(titles, abs) if t.strip() != '' and p.strip() != '']

        texts = ai_data + cv_data + lg_data
        random.shuffle(texts)
        print('Done.')
        train_text = texts[:int(len(texts) * 0.9)]
        val_text = texts[int(len(texts) * 0.9):int(len(texts) * 0.95)]
        test_text = texts[int(len(texts) * 0.95):]

        if make_train:
            train_preproc = Preprocessor(tokenizer, train_seq_len, data_type)
            d_train = ArxivDataset(train_text, train_preproc)
            print('Train dataset size', len(d_train))
            loaders.append(data.DataLoader(d_train,
                                           # sampler=DistributedSampler(d_train) if distributed else None,
                                           batch_size=train_bsz,
                                           pin_memory=True,
                                           drop_last=True,
                                           num_workers=num_workers,
                                           collate_fn=train_collate_fn) if d_train else None)
        if make_val:
            val_preproc = Preprocessor(tokenizer, val_seq_len, data_type)
            d_val = ArxivDataset(val_text, val_preproc)
            print('Val dataset size', len(d_val))
            loaders.append(data.DataLoader(d_val,
                                           # sampler=DistributedSampler(d_val),
                                           batch_size=val_bsz,
                                           pin_memory=True,
                                           drop_last=True,
                                           num_workers=num_workers,
                                           collate_fn=val_collate_fn) if d_val else None)
        if make_test:
            test_preproc = Preprocessor(tokenizer, test_seq_len, data_type)
            d_test = ArxivDataset(test_text, test_preproc)
            print('Test dataset size', len(d_test))
            loaders.append(data.DataLoader(d_test,
                                           # sampler=DistributedSampler(d_val),
                                           batch_size=test_bsz,
                                           pin_memory=True,
                                           drop_last=True,
                                           num_workers=num_workers,
                                           collate_fn=test_collate_fn) if d_test else None)
    elif dataset_name == 'yp':
        train_collate_fn = collate_fn
        val_collate_fn = collate_fn
        test_collate_fn = collate_fn

        if make_train:
            train_preproc = Preprocessor(tokenizer, train_seq_len, data_type)
            d_train = YelpDataset(os.path.join(data_dir, 'yelp/yelp.train.txt'), train_preproc)
            print('Train dataset size', len(d_train))
            loaders.append(data.DataLoader(d_train,
                                           # sampler=DistributedSampler(d_train) if distributed else None,
                                           batch_size=train_bsz,
                                           pin_memory=True,
                                           drop_last=True,
                                           num_workers=num_workers,
                                           collate_fn=train_collate_fn) if d_train else None)
        if make_val:
            val_preproc = Preprocessor(tokenizer, val_seq_len, data_type)
            d_val = YelpDataset(os.path.join(data_dir, 'yelp/yelp.valid.txt'), val_preproc)
            print('Val dataset size', len(d_val))
            loaders.append(data.DataLoader(d_val,
                                           # sampler=DistributedSampler(d_val),
                                           batch_size=val_bsz,
                                           pin_memory=True,
                                           drop_last=True,
                                           num_workers=num_workers,
                                           collate_fn=val_collate_fn) if d_val else None)
        if make_test:
            test_preproc = Preprocessor(tokenizer, test_seq_len, data_type)
            d_test = YelpDataset(os.path.join(data_dir, 'yelp/yelp.test.txt'), test_preproc)
            print('Test dataset size', len(d_test))
            loaders.append(data.DataLoader(d_test,
                                           # sampler=DistributedSampler(d_val),
                                           batch_size=test_bsz,
                                           pin_memory=True,
                                           drop_last=True,
                                           num_workers=num_workers,
                                           collate_fn=test_collate_fn) if d_test else None)
    else:
        raise Exception('Invalid dataset')

    return loaders