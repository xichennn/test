# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from datamodules.carla_datamodule import CarlaDataModule
from models.hivt import HiVT

if __name__ == '__main__':
    pl.seed_everything(2022)

    parser = ArgumentParser()
    # parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--root', type=str, default="/content/drive/MyDrive/carla_data/")
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--val_batch_size', type=int, default=2)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--log_every_n_steps', type=int, default=2)
    parser.add_argument('--monitor', type=str, default='val_minFDE', choices=['val_minADE', 'val_minFDE', 'val_minMR'])
    parser.add_argument('--save_top_k', type=int, default=5)
    parser = HiVT.add_model_specific_args(parser)
    args = parser.parse_args()

    datamodule = CarlaDataModule.from_argparse_args(args)
    model_checkpoint = ModelCheckpoint(monitor=args.monitor, save_top_k=args.save_top_k, mode='min')
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[model_checkpoint], log_every_n_steps=2)
    model = HiVT(**vars(args))
    # model = HiVT.load_from_checkpoint(checkpoint_path="lightning_logs/version_31/checkpoints/epoch=9-step=240.ckpt", parallel=True)
    trainer.fit(model, datamodule)
