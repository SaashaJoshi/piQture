import numpy as np
from PIL import Image
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

from piqture.embeddings.image_embeddings.brqi import BRQI

def load_and_preprocess_image(image_path, target_size=(4, 4)):
    # Load the image
    img = Image.open(image_path)
    
    # Convert to grayscale
    img = img.convert('L')
    
    # Resize the image
    img = img.resize(target_size)
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Normalize pixel values to range [0, 255]
    img_array = (img_array / img_array.max() * 255).astype(int)
    
    return img_array

def create_brqi_circuit(img_array):
    img_dims = img_array.shape
    pixel_vals = [img_array.flatten().tolist()]
    
    # Create BRQI object
    brqi_object = BRQI(img_dims, pixel_vals)
    
    # Generate BRQI circuit
    brqi_circuit = brqi_object.brqi()
    
    return brqi_circuit

def visualize_results(original_image, brqi_circuit):
    # Display original image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')
    
    # Simulate the quantum circuit
    backend = Aer.get_backend('qasm_simulator')
    transpiled_circuit = transpile(brqi_circuit, backend)
    job = backend.run(transpiled_circuit, shots=1000)
    result = job.result()
    counts = result.get_counts()
    
    # Plot the histogram of measurement results
    plt.subplot(1, 2, 2)
    plot_histogram(counts)
    plt.title('BRQI Measurement Results')
    
    plt.tight_layout()
    plt.show()

def main():
    # Path to your input image
    image_path = '/path/to/image.jpg'
    
    # Load and preprocess the image
    img_array = load_and_preprocess_image(image_path)
    
    # Create BRQI circuit
    brqi_circuit = create_brqi_circuit(img_array)
    
    # Visualize results
    visualize_results(img_array, brqi_circuit)

if __name__ == "__main__":
    main()
