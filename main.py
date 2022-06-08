import pandas as pd

from utils.lr_finder import custom_lr_finder
from utils.config_reader import cfg_load
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from model.bart_model import KangBart
from utils.common.project_paths import GetPaths
from dataload.dataloader import LitDataLoader
from sklearn.model_selection import train_test_split


def main():
    # configs
    config = cfg_load()
    
    # 모델 및 토크나이저
    model = KangBart()  # 모델
    # model.apply_ckpt('./ckpt/model.ckpt')
    tokenizer = model.tokenizer  # 토크나이저

    # 데이터 프레임
    csv = pd.read_csv(GetPaths.get_data_folder(config['csv']), delimiter='\t')  # tsv 읽어오기

    train_df, valid_df = train_test_split(csv, config['train_test_split']['test_size'])  # train, valid 나누기
    train_df.reset_index(drop=True, inplace=True)  # Index 리셋
    valid_df.reset_index(drop=True, inplace=True)  # Index 리셋

    # 데이터 로더
    lit_loaders = LitDataLoader(train_df=train_df,
                                valid_df=valid_df,
                                tokenizer=tokenizer,
                                translator=False)

    train_loader = lit_loaders.train_dataloader(**config['train_loader'])  # 학습 Data Loader
    valid_loader = lit_loaders.val_dataloader(**config['valid_loader'])  # 검증 Data Loader

    # Logger
    wandb_logger = WandbLogger(project='kang_bart')

    # 콜백
    ckpt_callback = ModelCheckpoint(**config['ModelCheckpoint'])

    early_stop = EarlyStopping(**config['EarlyStopping'])  # early stopping
    lr_monitor = LearningRateMonitor()  # logger에 lr 추가

    # 학습
    trainer = Trainer(max_epochs=config['trainer']['max_epochs'],
                      accelerator=config['trainer']['accelerator'],
                      gpus=config['trainer']['gpus'],
                      logger=wandb_logger,
                      callbacks=[ckpt_callback, early_stop, lr_monitor],
                      precision=config['trainer']['precision']
                      )

    wandb_logger.watch(model)
    trainer.fit(model, train_loader, valid_loader)


if __name__ == '__main__':
    main()