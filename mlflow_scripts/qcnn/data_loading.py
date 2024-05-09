import mlflow
import click
import piqture
from piqture.data_loader import load_mnist_dataset


@click.command()
@click.option("--img_size", default=28, help="size to which images will be resized.")
@click.option("--batch_size", default=None, help="batch size for the dataset.")
@click.option("--labels", default=None, help="list of desired labels.")
@click.option("--norm_min", default=None, help="minimum value for normalization.")
@click.option("--norm_max", default=None, help="maximum value for normalization.")
def load_data(img_size, batch_size, labels, norm_min, norm_max):
    with mlflow.start_run():
        train_dataloader, test_dataloader = load_mnist_dataset(
            img_size=img_size,
            batch_size=batch_size,
            labels=labels,
            normalize_min=norm_min,
            normalize_max=norm_max,
        )

        for batch in train_dataloader:
            if batch:
                print(batch)


if __name__ == "__main__":
    load_data()
