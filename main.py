import pandas as pd
from utils.lr_finder import custom_lr_finder
from utils.config_reader import cfg_load
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from model.bart_model import Bart
from utils.common.project_paths import GetPaths
from dataload.dataloader import LitDataLoader
from sklearn.model_selection import train_test_split


def main():
    # 모델 및 토크나이저

    config = cfg_load()
    model = Bart()  # 모델
    # model.apply_ckpt('./ckpt/model.ckpt')
    tokenizer = model.tokenizer  # 토크나이저

    # 데이터 프레임
    csv = pd.read_csv(GetPaths.get_data_folder('train.tsv'), delimiter='\t')  # tsv 읽어오기

    train_df, valid_df = train_test_split(csv, test_size=0.1)  # train, valid 나누기
    train_df.reset_index(drop=True, inplace=True)  # Index 리셋
    valid_df.reset_index(drop=True, inplace=True)  # Index 리셋

    # 데이터 로더
    lit_loaders = LitDataLoader(train_df=train_df,
                                valid_df=valid_df,
                                tokenizer=tokenizer,
                                translator=False)

    train_loader = lit_loaders.train_dataloader(num_workers=4)  # 학습 Data Loader
    valid_loader = lit_loaders.val_dataloader(num_workers=4)  # 검증 Data Loader

    # Logger
    wandb_logger = WandbLogger(project='kang_bart')

    # 콜백
    ckpt_callback = ModelCheckpoint(dirpath='./ckpt',
                                    filename='./model',
                                    monitor='valid_loss',
                                    save_top_k=1,
                                    save_weights_only=True,
                                    mode='min',
                                    save_last=False,
                                    verbose=True)

    early_stop = EarlyStopping(monitor='valid_loss', verbose=True, patience=10, mode='min')  # early stopping
    lr_monitor = LearningRateMonitor()  # logger에 lr 추가

    # 학습률 찾기 및 학습률 갱신
    # new_lr = custom_lr_finder(model, train_loader)
    # model.lr = new_lr

    # 학습
    trainer = Trainer(max_epochs=100,
                      accelerator='gpu',
                      gpus=1,
                      logger=wandb_logger,
                      callbacks=[ckpt_callback, early_stop, lr_monitor],
                      precision=16
                      )

    wandb_logger.watch(model)
    trainer.fit(model, train_loader, valid_loader)


if __name__ == '__main__':
    main()