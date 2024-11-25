from pprint import pprint

import torch
import wandb
import torchmetrics
import lightning as pl

from torch.utils.data import DataLoader

from ml.util import REGISTRY
from ml.optim import get_loss
from ml.trainer import BaseTrainer, ODETrainer, MyWandBLogger


def train_recon(cfg, fast_dev_run=False):
    logger = MyWandBLogger(
        name=cfg.run_name, project=cfg.project, log_model='all', id=cfg.run_id, allow_val_change=True
    )
    logger.experiment
    model_checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
        dirpath=wandb.run.dir if logger.experiment else None,
        monitor=cfg.training.metric_to_monitor,
        mode=cfg.training.metric_mode,
        save_top_k=cfg.training.save_top_k,
        save_last=cfg.training.save_last,
        filename='model_best',
    )
    trainer = pl.Trainer(
        logger=logger,
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
        log_every_n_steps=1,
    )

    dataset_class = REGISTRY['dataset'].get(cfg.dataset.name)

    train_ds = dataset_class(root=cfg.data_dir, **cfg.dataset.train)
    val_ds = dataset_class(root=cfg.data_dir, **cfg.dataset.val)

    if 'test' in cfg.dataset:
        test_ds = dataset_class(root=cfg.data_dir, **cfg.dataset.test)
    else:
        test_ds = None

    t_dl, v_dl = [
        DataLoader(
            dataset=ds, batch_size=cfg.batch_size, num_workers=num_w, shuffle=shuffle,
            pin_memory=True, persistent_workers=True, drop_last=skip_last,
            generator=torch.Generator().manual_seed(cfg.training.dataloader_seed) if shuffle else None
        )
        for ds, num_w, shuffle, skip_last in [(train_ds, cfg.num_workers, True, True), (val_ds, 1, False, False)]
    ]

    test_dl = DataLoader(
        dataset=test_ds, batch_size=cfg.batch_size, num_workers=1, shuffle=False,
        pin_memory=True, persistent_workers=True, drop_last=False,
    ) if test_ds else None

    # handle varying batch sizes and training steps so that it does not have to be hardcoded
    for _lr_n, _lr_p, _lr_i in zip(cfg.lr_scheduler.name, cfg.lr_scheduler.params, cfg.lr_scheduler.interval):
        if _lr_n == 'get_cosine_schedule_with_warmup':
            _lr_p['num_warmup_steps'] = int(_lr_p['num_warmup_steps'] * len(t_dl))
            _lr_p['num_training_steps'] = int(_lr_p['num_training_steps'] * len(t_dl)) + 1

    wandb.config.update(cfg.to_dict())

    pprint(cfg.to_dict())

    print(REGISTRY)

    loss_fn = get_loss(cfg.loss_fn.name, cfg.loss_fn.params)
    metrics = {m.short_name: getattr(torchmetrics, m.name)(**m.params) for m in cfg.metrics}

    recon_training_module_cls = BaseTrainer
    ae = REGISTRY['model'].get(cfg.model.name)(**cfg.model.params)
    training_module = recon_training_module_cls(
        cfg, ae, loss_fn, metrics, dataset_class.NORMALIZE, dataset_class.DE_NORMALIZE
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

    return ae


def train_ode(cfg, ae, fast_dev_run=False):
    logger = MyWandBLogger(
        name=cfg.run_name, project=cfg.project, log_model='all', id=cfg.run_id, allow_val_change=True
    )
    logger.experiment
    model_checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
        dirpath=wandb.run.dir if logger.experiment else None,
        monitor=cfg.training.metric_to_monitor,
        mode=cfg.training.metric_mode,
        save_top_k=cfg.training.save_top_k,
        save_last=cfg.training.save_last,
        filename='model_best',
    )
    trainer = pl.Trainer(
        logger=logger,
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
        log_every_n_steps=1,
    )

    dataset_class = REGISTRY['dataset'].get(cfg.dataset.name)

    train_ds = dataset_class(root=cfg.data_dir, **cfg.dataset.train)
    val_ds = dataset_class(root=cfg.data_dir, **cfg.dataset.val)

    if 'test' in cfg.dataset:
        test_ds = dataset_class(root=cfg.data_dir, **cfg.dataset.test)
    else:
        test_ds = None

    t_dl, v_dl = [
        DataLoader(
            dataset=ds, batch_size=cfg.batch_size, num_workers=num_w, shuffle=shuffle,
            pin_memory=True, persistent_workers=True, drop_last=skip_last,
            generator=torch.Generator().manual_seed(cfg.training.dataloader_seed) if shuffle else None
        )
        for ds, num_w, shuffle, skip_last in [(train_ds, cfg.num_workers, True, True), (val_ds, 1, False, False)]
    ]

    test_dl = DataLoader(
        dataset=test_ds, batch_size=cfg.batch_size, num_workers=1, shuffle=False,
        pin_memory=True, persistent_workers=True, drop_last=False,
    ) if test_ds else None

    # handle varying batch sizes and training steps
    for _lr_n, _lr_p, _lr_i in zip(cfg.lr_scheduler.name, cfg.lr_scheduler.params, cfg.lr_scheduler.interval):
        if _lr_n == 'get_cosine_schedule_with_warmup':
            _lr_p['num_warmup_steps'] = int(_lr_p['num_warmup_steps'] * len(t_dl))
            _lr_p['num_training_steps'] = int(_lr_p['num_training_steps'] * len(t_dl)) + 1

    wandb.config.update(cfg.to_dict())

    pprint(cfg.to_dict())

    loss_fn = get_loss(cfg.loss_fn.name, cfg.loss_fn.params)
    metrics = {m.short_name: getattr(torchmetrics, m.name)(**m.params) for m in cfg.metrics}

    ode_training_module_cls = ODETrainer
    neural_ode = REGISTRY['model'].get(cfg.model.name)(**cfg.model.params)
    training_module = ode_training_module_cls(
        cfg, neural_ode, loss_fn, metrics, dataset_class.NORMALIZE, dataset_class.DE_NORMALIZE,
        ae.encoder, ae.decoder
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

    os.environ['WANDB_MODE'] = 'disabled'
    # os.environ["HTTPS_PROXY"] = 'http://proxy.utwente.nl:3128'

    split_seed = int.from_bytes(os.urandom(4), 'little')
    lightning_seed = int.from_bytes(os.urandom(4), 'little')
    dataloader_seed = int.from_bytes(os.urandom(4), 'little')

    for (_dataset, epochs) in [('HeLa', 25)]:
        num_channels = 1 if _dataset in ['HeLa'] else 3
        b_s = 64
        wd_recon = 1e-4
        wd_ode = 1e-6

        run_name = _dataset

        _cfg = {
            'seed': lightning_seed,
            'run_name': f'{run_name}_recon',
            'project': 'HeLa',
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
                'train': {'split': 'train', 'seed': split_seed},
                'val': {'split': 'val', 'seed': split_seed},
            },
            'data_dir': os.path.join('.', 'data'),
            'batch_size': b_s,
            'num_workers': 2,
            'model': {
                'name': f'{_dataset}-ae',
                'params': {}
            },
            'loss_fn': {
                'name': 'MSELoss', 'params': {}
            },
            'metrics': [
                ConfigDict({
                    'name': 'MeanSquaredError', 'short_name': 'mse',
                    'params': {'squared': True}
                }),
            ],
            'opt': {
                'turn_off_norm_weight_decay': True,
                'name': 'SGD', 'params': {'lr': 0.1, 'momentum': 0.9, 'weight_decay': wd_recon, 'nesterov': True},
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
        ae = train_recon(_cfg, fast_dev_run=False)
        wandb.finish()

        _cfg = {
            'seed': lightning_seed,
            'run_name': f'{run_name}_ode',
            'project': 'HeLa',
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
                'train': {'split': 'train', 'seed': split_seed},
                'val': {'split': 'val', 'seed': split_seed},
            },
            'data_dir': os.path.join('.', 'data'),
            'batch_size': b_s,
            'num_workers': 2,
            'model': {
                'name': f'{_dataset}-ode',
                'params': {'dim': 512, 'timesteps': 1}
            },
            'loss_fn': {
                'name': 'MSELoss', 'params': {}
            },
            'metrics': [
                ConfigDict({
                    'name': 'MeanSquaredError', 'short_name': 'mse',
                    'params': {'squared': True}
                }),
            ],
            'opt': {
                'turn_off_norm_weight_decay': True,
                'name': 'AdamW',
                'params': {'lr': 1e-3, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': wd_ode, 'amsgrad': False},
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
        train_ode(_cfg, ae, fast_dev_run=False)
        wandb.finish()
