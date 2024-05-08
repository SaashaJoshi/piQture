import mlflow
import click
from piqture.data_loader.mnist_data_loader import load_mnist_dataset

@click.command()
@click.option("--labels", default=None, help="list of desired labels.")
@click.option("--batch_size", default=None, help="batch size for the dataset.")
@click.option("--img_size", default=2, help="size to which images will be resized.")
@click.option("--norm_min", default=0, help="minimum value for normalization.")
@click.option("--norm_max", default=1, help="maximum value for normalization.")
def load_data(labels, batch_size, img_size, norm_min, norm_max):
    with mlflow.start_run():
        train_dataloader, test_dataloader = load_mnist_dataset(labels, batch_size, img_size, norm_min, norm_max)
        print(train_dataloader)


if __name__ == "__main__":
    load_data()
