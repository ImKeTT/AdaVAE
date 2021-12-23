import pandas as pd
from torch.utils.data import Dataset


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
