import torch
from torch.utils.data import Dataset
from utils.text_infilling import TextInfilling
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512, train_stage=True):
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.train_stage = train_stage
        self.max_len = max_len
        self.text_infilling = TextInfilling()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        instance = self.df.iloc[idx]

        # input_ids
        tokenized_input = self.tokenizer.encode(instance['원문'])
        masked_input = self.text_infilling.mask(tokenized_input)

        input_ids = np.concatenate([[self.tokenizer.bos_token_id], masked_input, [self.tokenizer.eos_token_id]])
        input_ids = self.add_padding_data(input_ids)

        # label_ids
        label_ids = self.tokenizer.encode(instance['원문'])
        label_ids = np.concatenate([[self.tokenizer.bos_token_id], label_ids, [self.tokenizer.eos_token_id]])

        # decoder input
        dec_input_ids = np.concatenate([[self.tokenizer.eos_token_id], [self.tokenizer.bos_token_id], label_ids[:-1]])
        dec_input_ids = self.add_padding_data(dec_input_ids)

        label_ids = self.add_padding_data(label_ids)  # label에 padding

        return {'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'decoder_input_ids': torch.tensor(dec_input_ids, dtype=torch.long),
                'labels': torch.tensor(label_ids, dtype=torch.long)}

    def add_padding_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.tokenizer.pad_token_id] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]
        return inputs


def test():
    import pandas as pd
    from utils.common.project_paths import GetPaths
    from custom_tokenizer.custom_tokenizer import get_tokenizer
    from torch.utils.data import DataLoader
    df = pd.read_csv(GetPaths.get_data_folder('train.tsv'), delimiter='\t')
    tokenizer = get_tokenizer()
    w = 'ひ'
    a = tokenizer.encode(w)
    b = tokenizer.decode(a)
    d_set = DataLoader(CustomDataset(df, tokenizer))
    for i in d_set:
        print(i)


if __name__ == '__main__':
    test()
