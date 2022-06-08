from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from dataload.dataset import CustomDataset


class LitDataLoader(LightningDataModule):
    def __init__(self, train_df=None, valid_df=None, tokenizer=None):
        super().__init__()
        self.train_df = train_df
        self.valid_df = valid_df
        self.tokenizer = tokenizer

    def __create_dataset(self, mode="train"):
        if mode == "train":
            return CustomDataset(
                df=self.train_df, tokenizer=self.tokenizer, train_stage=True
            )
        elif mode == "valid":
            return CustomDataset(
                df=self.valid_df, tokenizer=self.tokenizer, train_stage=True
            )
        else:
            raise Exception("mode should be in [train, valid]")

    def train_dataloader(
        self, batch_size=4, shuffle=True, drop_last=False, num_workers=4
    ):
        dataset = self.__create_dataset("train")
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )

    def val_dataloader(
        self, batch_size=4, shuffle=False, drop_last=False, num_workers=4
    ):
        dataset = self.__create_dataset("valid")
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )


if __name__ == "__main__":
    tokenizer = "Load Tokenizer"
    train_df = "Load train csv"
    valid_df = "Load valid csv"
    data_loader = LitDataLoader(
        tokenizer=tokenizer, train_df=train_df, valid_df=valid_df
    )
