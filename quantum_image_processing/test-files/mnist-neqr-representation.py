import torchvision
import torch.utils.data
from quantum_image_processing.data_encoder.image_representations.neqr import NEQR
from quantum_image_processing.data_loader.mnist_data_loader import load_mnist_data
import matplotlib.pyplot as plt

if __name__ == "__main__":
    _, __, train, test = load_mnist_data()
    train_data = iter(train)
    train_img, train_labels = next(train_data)

    print(train_img.shape, train_labels.shape)

    torch.set_printoptions(precision=0)

    # plt.hist(np.array(train_img[i]).ravel(), bins=50, density=True);
    # plt.xlabel("pixel values")
    # plt.ylabel("relative frequency")
    # plt.title("distribution of pixels")
    # plt.show()

    normal = torchvision.transforms.Normalize(mean=0.0, std=(1 / 255.0))(train_img[0])

    new_dim = normal.reshape(-1, 4)
    new_dim_list = new_dim.tolist()

    color_vals = new_dim_list[0]
    image_size = (2, 2)

    circ = NEQR(image_size, color_vals)
    circ = circ.image_encoding(measure=True)

    # Measurement results
    counts = NEQR.get_simulator_result(
        circ,
        backend="qasm_simulator",
        shots=1024,
        plot_counts=True,
    )
