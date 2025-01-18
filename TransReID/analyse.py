import umap
import numpy as np
import matplotlib.pyplot as plt
import json 
import os 

# Assuming feats is already defined with keys 0 to 14
# feats = {0: [...], 1: [...], ..., 14: [...]}

import torch
import torch.nn.functional as F
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import random
import numpy as np
from scipy.spatial import distance

def knn_classify(training_data, test_vector, k=3):
    """
    Perform k-Nearest Neighbors classification.

    Args:
        training_data (dict): Dictionary with class labels as keys and lists of feature vectors as values.
        test_vector (list or np.ndarray): The feature vector to classify.
        k (int): Number of neighbors to consider.

    Returns:
        str: Predicted class label.
    """
    # Flatten the training data into a list of tuples: (vector, label)
    flattened_data = []
    for label, vectors in training_data.items():
        for vector in vectors:
            flattened_data.append((np.array(vector), label))
    
    # Compute distances from the test vector to all training vectors
    distances = [(distance.euclidean(test_vector, vec), label) for vec, label in flattened_data]

    # Sort by distance and select the k closest
    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]

    # Perform majority vote on the k nearest labels
    class_votes = {}
    for _, label in k_nearest:
        class_votes[label] = class_votes.get(label, 0) + 1

    # Return the class with the most votes
    predicted_label = max(class_votes, key=class_votes.get)
    return predicted_label



# Gaussian kernel using PyTorch for GPU computation
def gaussian_kernel(x, y, sigma=1.0):
    # Squared Euclidean distance using PyTorch
    dist = torch.cdist(x, y, p=2) ** 2
    return torch.exp(-dist / (2 * sigma**2))

# Compute MMD using PyTorch
def compute_mmd(X, Y, sigma=0.5):
    XX = gaussian_kernel(X, X, sigma)
    YY = gaussian_kernel(Y, Y, sigma)
    XY = gaussian_kernel(X, Y, sigma)
    return torch.mean(XX) + torch.mean(YY) - 2 * torch.mean(XY)

def compute_cosine(X, Y):
    # Assuming X and Y are your input tensors of shape [B, D]
    # B is the batch size, D is the dimensionality of the vectors

    # Normalize the vectors to have unit length along the dimension D
    X_normalized = F.normalize(X, p=2, dim=1)  # Shape [B, D]
    Y_normalized = F.normalize(Y, p=2, dim=1)  # Shape [B, D]

    # Compute the cosine similarities between each pair of vectors in the batch
    cosine_similarities = torch.mm(X_normalized, Y_normalized.T)  # Shape [B, B]
    return cosine_similarities.mean()

# Load your feats dictionary and convert it to PyTorch tensors (assuming you've done this step)
# Example: feats = {key: torch.tensor(data).cuda() for key, data in feats.items()}

def make_mmd_plot(feats, output_file):

    # Ensure everything is on the GPU (assuming you have a GPU available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Converting to tensors")
    # Convert the list of vectors to PyTorch tensors and move them to the GPU
    feats1, feats2 = {}, {}
    for key in feats:
        # Convert each list of vectors (e.g., numpy array or list) to a PyTorch tensor
        # and ensure it's in the correct shape (e.g., (num_samples, D))
        #random.shuffle(feats[key])
        random.shuffle(feats[key])
        feats1[key] = torch.tensor(feats[key][:500], dtype=torch.float32).to(device)
        random.shuffle(feats[key])
        feats2[key] = torch.tensor(feats[key][:500], dtype=torch.float32).to(device)
    #breakpoint()
    # Number of keys
    n_keys = len(feats1)
    
    # Initialize matrix to store MMD scores
    mmd_matrix = np.zeros((n_keys, n_keys))
    print("Computing MMD")
    # Compute the MMD between each pair of keys using PyTorch
    for i, key1 in enumerate(feats):
        X = feats1[key1].to(device)  # Ensure the tensor is on the GPU
        for j, key2 in enumerate(feats):
            Y = feats2[key2].to(device)  # Ensure the tensor is on the GPU
            #breakpoint()
            mmd_matrix[i, j] = compute_mmd(X, Y).item() * 1000  # Compute MMD and store result
    print("Saving MMD")
    # Plotting the heatmap
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(mmd_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, xticklabels=feats.keys(), yticklabels=feats.keys())
    plt.title("MMD Score Matrix (Values scaled up by 1000)")
    # Save the plot to the specified output file
    plt.savefig(output_file, format='png')
    plt.close()

def make_cosine_plot(feats, output_file):
    # Ensure everything is on the GPU (assuming you have a GPU available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Converting to tensors")
    # Convert the list of vectors to PyTorch tensors and move them to the GPU
    feats1, feats2 = {}, {}
    for key in feats:
        # Convert each list of vectors (e.g., numpy array or list) to a PyTorch tensor
        # and ensure it's in the correct shape (e.g., (num_samples, D))
        #random.shuffle(feats[key])
        random.shuffle(feats[key])
        feats1[key] = torch.tensor(feats[key][:500], dtype=torch.float32).to(device)
        random.shuffle(feats[key])
        feats2[key] = torch.tensor(feats[key][:500], dtype=torch.float32).to(device)
    #breakpoint()
    # Number of keys
    n_keys = len(feats1)
    
    # Initialize matrix to store MMD scores
    mmd_matrix = np.zeros((n_keys, n_keys))
    print("Computing cosine")
    # Compute the MMD between each pair of keys using PyTorch
    for i, key1 in enumerate(feats):
        X = feats1[key1].to(device)  # Ensure the tensor is on the GPU
        for j, key2 in enumerate(feats):
            Y = feats2[key2].to(device)  # Ensure the tensor is on the GPU
            #breakpoint()
            mmd_matrix[i, j] = compute_cosine(X, Y).item()  # Compute MMD and store result
    print("Saving cosine")
    # Plotting the heatmap
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(mmd_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, xticklabels=feats.keys(), yticklabels=feats.keys())
    plt.title("Cosine Similarity Score Matrix")
    # Save the plot to the specified output file
    plt.savefig(output_file, format='png')
    plt.close()

def plot_umap_feats(feats, output_file, nn=15):
    # Initialize lists to hold the vectors and their corresponding labels
    all_vectors = []
    all_labels = []
    print("Loading data")
    # Iterate over each key in the feats dictionary
    for key, vectors in feats.items():
        all_vectors.extend(vectors)  # Add all vectors from this key
        all_labels.extend([key] * len(vectors))  # Label them with the current key

    # Convert the list of vectors into a numpy array
    all_vectors = np.array(all_vectors)
    all_labels = np.array(all_labels).astype(int)  # Ensure labels are integers
    print("Loaded all data..processing umap")
    # Apply UMAP to reduce dimensions to 2D
    reducer = umap.UMAP(n_neighbors=nn, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(all_vectors)
    print("Processing umap done...saving")
    # Plot the UMAP embedding
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=all_labels, cmap='tab20', s=10, alpha=0.8)
    plt.colorbar(scatter, ticks=range(nn))  # Ensure 15 ticks for the 15 labels
    plt.title("UMAP Projection of Feature Vectors")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")

    # Save the plot to the specified output file
    plt.savefig(output_file, format='png')
    plt.close()

# Example usage
# Assuming feats is your dictionary of vectors
# plot_umap_feats(feats, "umap_projection.png")
# with open("/export/livia/home/vision/Rbhattacharya/work/CLIP-ReID/outputs/test/features_5cams.json", 'r') as f:
#     file_content = f.read()  # Read the entire content of the file as a string
#     feats = json.loads(file_content)  # Parse the JSON from the string

# plot_umap_feats(feats, "outputs/test/umap_5cams.png")

# with open("/export/livia/home/vision/Rbhattacharya/work/CLIP-ReID/outputs/test/features_fullmodel.json", 'r') as f:
#     file_content = f.read()  # Read the entire content of the file as a string
#     feats = json.loads(file_content)  # Parse the JSON from the string
#     #breakpoint()
#     feats = {key: value for key, value in feats.items() if len(value) > 0}

# make_cosine_plot(feats, "outputs/test/cosine_full_subsample.png")
# make_mmd_plot(feats, "outputs/test/mmd_full_subsample.png")

# with open("/export/livia/home/vision/Rbhattacharya/work/CLIP-ReID/outputs/test/features_10cams.json", 'r') as f:
#     file_content = f.read()  # Read the entire content of the file as a string
#     feats = json.loads(file_content)  # Parse the JSON from the string
#     #breakpoint()
#     feats = {key: value for key, value in feats.items() if len(value) > 0}

# make_cosine_plot(feats, "outputs/test/cosine_full_subsample_10.png")
#make_mmd_plot(feats, "outputs/test/mmd_full_subsample_10.png")

# with open("/export/livia/home/vision/Rbhattacharya/work/CLIP-ReID/outputs/test/features_5cams.json", 'r') as f:
#     file_content = f.read()  # Read the entire content of the file as a string
#     feats = json.loads(file_content)  # Parse the JSON from the string
#     #breakpoint()
#     feats = {key: value for key, value in feats.items() if len(value) > 0}

# make_cosine_plot(feats, "outputs/test/cosine_full_subsample_5.png")
#make_mmd_plot(feats, "outputs/test/mmd_full_subsample_5.png")

# with open("/export/livia/home/vision/Rbhattacharya/work/CLIP-ReID/outputs/test/dukefeatures_full.json", 'r') as f:
#     file_content = f.read()  # Read the entire content of the file as a string
#     feats = json.loads(file_content)  # Parse the JSON from the string
#     feats = {key: value for key, value in feats.items() if len(value) > 0}
# plot_umap_feats(feats, "outputs/test/duke_umap_full.png", nn=8)