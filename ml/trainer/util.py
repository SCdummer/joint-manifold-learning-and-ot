from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

try:
    import wandb
    from wandb.sdk.lib import RunDisabled
    from wandb.wandb_run import Run

    wandb.require("core")
except ModuleNotFoundError:
    # needed for test mocks, these tests shall be updated
    wandb, Run, RunDisabled = None, None, None


class MyWandBLogger(WandbLogger):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint) -> None:
        if self._log_model:
            if checkpoint_callback.best_model_path:
                # get the best model path
                best_model_path = checkpoint_callback.best_model_path
                # log the best model
                wandb.save(best_model_path, base_path=wandb.run.dir)


def get_best_ckpt(run_id, model_version):
    if run_id is not None:
        try:
            ckpt_path = wandb.restore(f'{model_version}.ckpt', root=wandb.run.dir, replace=True).name
        except wandb.errors.CommError:
            # if the run is disabled, we can't restore the model or run does not exist
            ckpt_path = None
        except ValueError:
            # if the checkpoint does not exist, we can't restore the model
            ckpt_path = None
    else:
        ckpt_path = None
    return ckpt_path
