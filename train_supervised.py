import time

import lightning as L
from lightning.pytorch.loggers import WandbLogger

from dataset import EuXFELCurrentDataModule
from supervised import SupervisedCurrentProfileInference


def main():
    data_module = EuXFELCurrentDataModule(batch_size=64, num_workers=5)
    model = SupervisedCurrentProfileInference(learning_rate=1e-3, negative_slope=0.01)

    wandb_logger = WandbLogger(project="virtual-diagnostics-euxfel-current-supervised")

    trainer = L.Trainer(
        max_epochs=5,
        logger=wandb_logger,
        accelerator="cpu",
        devices="auto",
        log_every_n_steps=50,
        fast_dev_run=True,
    )
    trainer.fit(model, data_module)

    time.sleep(10)


if __name__ == "__main__":
    main()
