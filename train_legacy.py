import time

from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger

from dataset import EuXFELCurrentDataModule
from legacy import SupervisedCurrentProfileInference


def main():
    config = {
        "batch_size": 64,
        "learning_rate": 1e-3,
        "max_epochs": 5,
    }

    wandb_logger = WandbLogger(
        project="virtual-diagnostics-euxfel-current-legacy", config=config
    )
    config = dict(wandb_logger.experiment.config)

    data_module = EuXFELCurrentDataModule(
        batch_size=config["batch_size"], num_workers=5, normalize=True
    )
    model = SupervisedCurrentProfileInference(
        learning_rate=config["learning_rate"],
    )

    trainer = Trainer(
        max_epochs=config["max_epochs"],
        logger=wandb_logger,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=50,
    )
    trainer.fit(model, data_module)

    time.sleep(10)


if __name__ == "__main__":
    main()
