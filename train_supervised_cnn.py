import time

import lightning as L
from lightning.pytorch.loggers import WandbLogger

from dataset import EuXFELCurrentDataModule
from supervised_cnn import CNNCurrentReconstructor


def main():
    data_module = EuXFELCurrentDataModule(batch_size=64, num_workers=5)
    model = CNNCurrentReconstructor(learning_rate=1e-3, leaky_relu_negative_slope=0.01)

    wandb_logger = WandbLogger(project="virtual-diagnostics-euxfel-current-supervised")

    # TODO Fix errors raised on accelerator="mps" -> PyTorch pull request merged
    trainer = L.Trainer(
        max_epochs=100,
        logger=wandb_logger,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=1,
        overfit_batches=1,
    )
    trainer.fit(model, data_module)

    time.sleep(10)


if __name__ == "__main__":
    main()
