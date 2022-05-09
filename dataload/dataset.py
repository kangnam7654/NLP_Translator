from torch.utils.data import Dataset
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256, pad_index=1, ignore_index=-100, train_stage=True):
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.train_stage = train_stage
        self.max_len = max_len
        self.pad_index = pad_index
        self.ignore_index = ignore_index

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        instance = self.df.iloc[idx]
        input_ids = self.tokenizer.encode(instance['원문'])
        input_ids = self.add_padding_data(input_ids)

        label_ids = self.tokenizer.encode(instance['번역문'])
        label_ids.append(self.tokenizer.eos_token_id)

        dec_input_ids = [self.pad_index]
        dec_input_ids += label_ids[:-1]
        dec_input_ids = self.add_padding_data(dec_input_ids)

        label_ids = self.add_ignored_data(dec_input_ids)

        return {'input_ids': np.array(input_ids),
                'decoder_input_ids': np.array(dec_input_ids),
                'labels': np.array(label_ids)}

    def add_padding_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.pad_index] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]
        return inputs

    def add_ignored_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.ignore_index] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]
        return inputs


class BaseCustomDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256, pad_index=1, ignore_index=-100, train_stage=True):
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.train_stage = train_stage
        self.max_len = max_len
        self.pad_index = pad_index
        self.ignore_index = ignore_index

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        instance = self.df.iloc[idx]
        input_ids = self.tokenizer.encode(instance['원문'])
        input_ids = self.add_padding_data(input_ids)

        label_ids = self.tokenizer.encode(instance['원문'])
        label_ids.append(self.tokenizer.eos_token_id)

        dec_input_ids = [self.pad_index]
        dec_input_ids += label_ids[:-1]
        dec_input_ids = self.add_padding_data(dec_input_ids)

        label_ids = self.add_ignored_data(dec_input_ids)

        return {'input_ids': np.array(input_ids),
                'decoder_input_ids': np.array(dec_input_ids),
                'labels': np.array(label_ids)}

    def add_padding_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.pad_index] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]
        return inputs

    def add_ignored_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.ignore_index] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]
        return inputs


if __name__ == '__main__':
    pass
