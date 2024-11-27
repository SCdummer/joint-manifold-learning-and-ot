from pathlib import Path
from pprint import pprint

import torch
import wandb
import torchmetrics
import lightning as pl

from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

from ml.util import REGISTRY
from ml.trainer import JointReconODETrainer, MyWandBLogger


def prepare_lightning(cfg, fast_dev_run):
    wandb_logger = MyWandBLogger(
        name=cfg.run_name, project=cfg.project, log_model='all', id=cfg.run_id, allow_val_change=True
    )
    wandb_logger.experiment
    if wandb.run:
        _id = wandb.run.id
    else:
        import uuid
        _id = uuid.uuid4().hex
    cfg.logging_path = Path('.', 'out', f'{_id}')
    cfg.logging_path.mkdir(parents=True, exist_ok=True)

    model_logging_path = cfg.logging_path / 'model'
    model_logging_path.mkdir(parents=True, exist_ok=True)

    model_checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
        dirpath=wandb.run.dir if wandb_logger.experiment else model_logging_path,
        monitor=cfg.training.metric_to_monitor,
        mode=cfg.training.metric_mode,
        save_top_k=cfg.training.save_top_k,
        save_last=cfg.training.save_last,
        filename='model_best',
    )
    trainer = pl.Trainer(
        logger=[
            wandb_logger,
            CSVLogger(save_dir=cfg.logging_path, name='logs')
        ],
        callbacks=[
            model_checkpoint_callback,
            pl.pytorch.callbacks.LearningRateMonitor(logging_interval='step'),
        ],
        accelerator=cfg.training.accelerator,
        precision=cfg.training.precision,
        devices=cfg.training.devices,
        strategy=cfg.training.strategy,
        num_sanity_val_steps=cfg.training.num_sanity_val_steps,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        benchmark=cfg.training.benchmark,
        max_epochs=cfg.training.max_epochs,
        fast_dev_run=fast_dev_run,
        log_every_n_steps=5,
        check_val_every_n_epoch=5,
    )
    return model_checkpoint_callback, trainer


def update_num_steps_lr_scheduler(cfg, t_dl):
    # handle varying batch sizes and training steps so that it does not have to be hardcoded
    for _lr_n, _lr_p, _lr_i in zip(cfg.lr_scheduler.name, cfg.lr_scheduler.params, cfg.lr_scheduler.interval):
        if _lr_n == 'get_cosine_schedule_with_warmup':
            _lr_p['num_warmup_steps'] = int(_lr_p['num_warmup_steps'] * len(t_dl))
            _lr_p['num_training_steps'] = int(_lr_p['num_training_steps'] * len(t_dl)) + 1
    wandb.config.update(cfg.to_dict())


def prepare_ds_and_dls(cfg, dataset_class):
    train_ds = dataset_class(root=cfg.data_dir, **cfg.dataset.train)
    val_ds = dataset_class(root=cfg.data_dir, **cfg.dataset.val)
    if 'test' in cfg.dataset:
        test_ds = dataset_class(root=cfg.data_dir, **cfg.dataset.test)
    else:
        test_ds = None

    t_sampler = torch.utils.data.WeightedRandomSampler(
        train_ds.get_sampling_weights(), 500, replacement=True
    )
    v_sampler = torch.utils.data.WeightedRandomSampler(
        val_ds.get_sampling_weights(), 16 * cfg.batch_size, replacement=True
    )

    t_dl, v_dl = [
        DataLoader(
            dataset=ds, batch_size=cfg.batch_size, num_workers=num_w, shuffle=shuffle,
            pin_memory=True, persistent_workers=True, drop_last=skip_last,
            generator=torch.Generator().manual_seed(cfg.training.dataloader_seed) if shuffle else None,
            collate_fn=dataset_class.get_collate_fn(), sampler=sampler
        )
        for ds, num_w, shuffle, skip_last, sampler in [
            (train_ds, cfg.num_workers, False, True, t_sampler), (val_ds, 1, False, False, v_sampler)
        ]
    ]
    test_dl = DataLoader(
        dataset=test_ds, batch_size=cfg.batch_size, num_workers=1, shuffle=False,
        pin_memory=True, persistent_workers=True, drop_last=False, collate_fn=dataset_class.get_collate_fn()
    ) if test_ds else None
    return t_dl, v_dl, test_dl


def train_joint_recon_and_latent_dynamics(cfg, fast_dev_run=False):
    model_checkpoint_callback, trainer = prepare_lightning(cfg, fast_dev_run)
    dataset_class = REGISTRY['dataset'].get(cfg.dataset.name)
    t_dl, v_dl, test_dl = prepare_ds_and_dls(cfg, dataset_class)
    update_num_steps_lr_scheduler(cfg, t_dl)

    pprint(cfg.to_dict())

    ae = REGISTRY['model'].get(cfg.recon_model.name)(**cfg.recon_model.params)
    neural_ode = REGISTRY['model'].get(cfg.latent_dynamics_model.name)(**cfg.latent_dynamics_model.params)
    metrics = {m.short_name: getattr(torchmetrics, m.name)(**m.params) for m in cfg.metrics}
    training_module = JointReconODETrainer(
        cfg, dataset_class.NORMALIZE, dataset_class.DE_NORMALIZE, metrics,
        neural_ode=neural_ode, encoder=ae.encoder, decoder=ae.decoder
    )

    trainer.fit(
        training_module,
        train_dataloaders=t_dl, val_dataloaders=v_dl,
        ckpt_path=cfg.training.ckpt_path
    )

    try:
        if test_dl:
            trainer.test(
                model=training_module, dataloaders=test_dl,
                ckpt_path=model_checkpoint_callback.best_model_path
            )
    except Exception as e:
        print(e)


if __name__ == '__main__':
    import os
    from ml_collections import ConfigDict

    os.environ['WANDB_MODE'] = 'offline'
    # os.environ["HTTPS_PROXY"] = 'http://proxy.utwente.nl:3128'

    split_seed = int.from_bytes(os.urandom(4), 'little')
    lightning_seed = int.from_bytes(os.urandom(4), 'little')
    dataloader_seed = int.from_bytes(os.urandom(4), 'little')

    for (_dataset, epochs) in [('HeLa', 100)]:
        n = 3
        num_channels = 1 if _dataset in ['HeLa'] else 3
        b_s = 64 // (n + 1)
        wd_recon = 1e-4
        wd_ode = 0

        run_name = _dataset

        _cfg = {
            'seed': lightning_seed,
            'run_name': f'{run_name}_joint',
            'project': f'{_dataset}',
            'run_id': None,
            'training': {
                'metric_to_monitor': 'val/loss', 'metric_mode': 'min',
                'save_top_k': 1, 'save_last': True,
                'accelerator': 'gpu', 'precision': '32-true',
                'devices': 1, 'accumulate_grad_batches': 1, 'strategy': 'auto',
                'num_sanity_val_steps': 0,
                'gradient_clip_val': 5,
                'benchmark': True,
                'max_epochs': epochs,
                'ckpt_path': None,
                'dataloader_seed': dataloader_seed,
            },
            'dataset': {
                'name': _dataset,
                'train': {'split': 'train', 'seed': split_seed, 'n_successive': n},
                'val': {'split': 'val', 'seed': split_seed, 'n_successive': n},
            },
            'data_dir': os.path.join('.', 'data'),
            'batch_size': b_s,
            'num_workers': 2,
            'recon_model': {
                'name': f'{_dataset}-ae',
                'params': {}
            },
            'latent_dynamics_model': {
                'name': f'{_dataset}-ode',
                'params': {'dim': 512, 'timesteps': 10}
            },
            'metrics': [
                # ConfigDict({
                #     'name': 'MeanSquaredError', 'short_name': 'mse',
                #     'params': {'squared': True}
                # }),
            ],
            'opt': {
                'turn_off_norm_weight_decay': True,
                'name': 'SGD', 'params': {'lr': 0.1, 'momentum': 0.9, 'weight_decay': 1e-4, 'nesterov': True},
                # 'name': 'Adam', 'params': {'lr': 5e-4, 'weight_decay': wd_ode},
            },
            'lr_scheduler': {
                'name': ['get_cosine_schedule_with_warmup', ],
                'params': [
                    {
                        'num_warmup_steps': 0, 'num_training_steps': epochs,
                        'lr_min': 1e-6, 'start_factor': 1e-6,
                    }
                ],
                'interval': ['step', ],
            }
        }
        _cfg = ConfigDict(_cfg, convert_dict=True)
        train_joint_recon_and_latent_dynamics(_cfg, fast_dev_run=False)
        wandb.finish()
