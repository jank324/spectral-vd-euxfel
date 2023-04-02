# Generator outputs bunch length and shape
# Discriminator is given bunch length and shape of real and generated example and has to
# learn to tell them apart as real or generated

# TODO Pass hidden architecture to models
# TODO Only ReLU not leaky ReLU in Generator accrding to DC GAN paper

import time

import lightning as L
from lightning.pytorch.loggers import WandbLogger

from dataset import EuXFELCurrentDataModule
from gan import WassersteinGANGP


def main():
    data_module = EuXFELCurrentDataModule(batch_size=64, num_workers=5)
    model = WassersteinGANGP(
        critic_iterations=5,
        lambda_gradient_penalty=10,
        learning_rate=1e-4,
        leaky_relu_negative_slope=0.2,
    )

    wandb_logger = WandbLogger(project="virtual-diagnostics-euxfel-current-gan")

    # TODO Fix errors raised on accelerator="mps" -> PyTorch pull request merged
    trainer = L.Trainer(
        max_epochs=3,
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
