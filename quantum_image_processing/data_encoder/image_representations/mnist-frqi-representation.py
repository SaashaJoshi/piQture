from frqi import FRQI
from quantum_image_processing.data_loader.mnist_data_loader import load_mnist_data


if __name__ == "__main__":
    _, __, train, test = load_mnist_data()
    train_data = iter(train)
    train_img, train_labels = next(train_data)

    print(train_img.shape, train_labels.shape)

    # Normalize in [0, pi/2] if already done during data loading.
    # normal = torchvision.transforms.Normalize((0.2052,), (0.4840,))(train_img[0])
    # plt.imshow(normal[0], cmap=plt.get_cmap('gray'))

    new_dim = train_img[0].reshape(-1, 4)
    new_dim_list = new_dim.tolist()

    color_vals = new_dim_list[0]
    print(color_vals)
    image_size = (2, 2)

    circ = FRQI(image_size, color_vals)
    circ = circ.frqi(measure=True)

    # Measurement results
    counts = FRQI.get_simulator_result(
        circ,
        backend="qasm_simulator",
        shots=1024,
        plot_counts=True,
    )
