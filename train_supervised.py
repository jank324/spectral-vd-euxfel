import time

from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger

from dataset import EuXFELCurrentDataModule
from supervised import SupervisedCurrentProfileInference


def main():
    config = {
        "batch_size": 64,
        "current_feature_maps": 8,
        "formfactor_feature_maps": 8,
        "learning_rate": 1e-3,
        "max_epochs": 5,
        "mlp_width": 64,
        "negative_slope": 0.01,
        "softplus_beta": 1,
    }

    wandb_logger = WandbLogger(
        project="virtual-diagnostics-euxfel-current-supervised", config=config
    )
    config = dict(wandb_logger.experiment.config)

    data_module = EuXFELCurrentDataModule(
        batch_size=config["batch_size"], num_workers=5, normalize=True
    )
    model = SupervisedCurrentProfileInference(
        learning_rate=config["learning_rate"],
        mlp_width=config["mlp_width"],
        formfactor_feature_maps=config["formfactor_feature_maps"],
        current_feature_maps=config["current_feature_maps"],
        negative_slope=config["negative_slope"],
        softplus_beta=config["softplus_beta"],
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
