import time

from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from dataset import EuXFELCurrentDataModule
from legacy import SupervisedCurrentProfileInference


def main():
    config = {
        "batch_normalization": False,
        "batch_size": 64,
        "hidden_activation": "relu",
        "hidden_layer_width": 100,
        "learning_rate": 1e-3,
        "max_epochs": 10_000,
        "num_hidden_layers": 3,
    }

    wandb_logger = WandbLogger(
        project="virtual-diagnostics-euxfel-current-legacy", config=config
    )
    config = dict(wandb_logger.experiment.config)

    data_module = EuXFELCurrentDataModule(
        batch_size=config["batch_size"], num_workers=0, normalize=True
    )
    model = SupervisedCurrentProfileInference(
        batch_normalization=config["batch_normalization"],
        hidden_activation=config["hidden_activation"],
        hidden_layer_width=config["hidden_layer_width"],
        learning_rate=config["learning_rate"],
        num_hidden_layers=config["num_hidden_layers"],
    )

    early_stopping_callback = EarlyStopping(
        monitor="validate/loss", mode="min", patience=10
    )

    trainer = Trainer(
        max_epochs=config["max_epochs"],
        logger=wandb_logger,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=50,
        callbacks=early_stopping_callback,
    )
    trainer.fit(model, data_module)

    time.sleep(10)


if __name__ == "__main__":
    main()
