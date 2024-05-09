import click
import mlflow
from piqture.data_encoder import AngleEncoding


@click.command()
@click.option("--img_dims", help="dimensions of the input images.")
def data_embedding(img_dims):
    with mlflow.start_run():
        img_dims = tuple(map(int, img_dims.split(",")))
        embedding = AngleEncoding(img_dims=img_dims)

        mlflow.log_params(
            {
                "Embedding circuit": embedding.circuit,
                "Embedding parameters": embedding.parameters,
            }
        )


if __name__ == "__main__":
    data_embedding()
