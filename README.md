# Clustering Unlabeled Sound Data

## Project Description
This project focuses on applying unsupervised machine learning techniques, specifically clustering, to an unlabeled dataset of sound recordings. The primary goals are to:
1.  Extract relevant features from audio data.
2.  Explore the necessity and benefits of dimensionality reduction (using PCA and t-SNE) for high-dimensional data visualization and clustering.
3.  Implement and compare two prominent clustering algorithms, K-Means and DBSCAN.
4.  Evaluate their performance using internal validation metrics (Silhouette Score, Davies-Bouldin Index) and visual interpretability.
5.  Analyze the challenges of clustering unlabeled, high-dimensional data and relate findings to real-world scenarios.

## Dataset
The dataset consists of unlabeled sound recordings, provided as `.wav` files. The path to the data is `/content/drive/MyDrive/Colab_Notebooks/data/unlabelled_sounds`.

## Feature Extraction
Mel-spectrogram features are extracted from each `.wav` file. Mel-spectrograms are commonly used in audio processing as they represent the short-time power spectrum of a sound, transformed onto the Mel scale, which is a perceptual scale of pitches judged by listeners to be equal in distance from one another. For this project, a 128-mel-band spectrogram is computed, and the mean across time is taken to create a compact, fixed-size feature vector for each audio file. These features are then standardized using `StandardScaler` to ensure all features contribute equally to the distance calculations in clustering algorithms.

## Dimensionality Reduction
Initial attempts to visualize the raw 128-dimensional features (using 2D scatter plots and pair plots) proved ineffective due to the high dimensionality, making it impossible to discern any underlying data structure or clusters. This highlighted the "curse of dimensionality" and the necessity of dimensionality reduction.

Two dimensionality reduction techniques were applied:
1.  **Principal Component Analysis (PCA):** A linear technique that transforms data into a new coordinate system where the greatest variance by any projection of the data lies on the first coordinate (called the first principal component), the second greatest variance on the second coordinate, and so on. We reduced the data to 3 components for visualization.
2.  **t-Distributed Stochastic Neighbor Embedding (t-SNE):** A non-linear technique well-suited for visualizing high-dimensional datasets. It focuses on preserving local relationships, meaning points that are close in the high-dimensional space remain close in the low-dimensional embedding. We also reduced the data to 3 components for visualization.

**Comparison of PCA vs. t-SNE for Cluster Separability:**
Based on the 3D visualizations, **t-SNE provided noticeably better separability of clusters compared to PCA**. The t-SNE plot displayed several more distinct and compact groupings of points with clear separation, whereas the PCA plot showed a more diffuse cloud of points without readily apparent clusters. This is because t-SNE is designed to preserve local structure and reveal intrinsic data groupings, while PCA focuses on global variance.

## Clustering Algorithms
Two clustering algorithms were implemented and compared:

1.  **K-Means:** A centroid-based, partition clustering algorithm that aims to partition `n` observations into `k` clusters in which each observation belongs to the cluster with the nearest mean (centroid).
    * **Optimal K:** The Elbow Method was used to determine the optimal number of clusters, suggesting `k=4` as the most suitable choice.
    * **Performance:** K-Means successfully identified 4 clusters, showing good performance with a positive Silhouette Score and a relatively low Davies-Bouldin Index.

2.  **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):** A density-based clustering algorithm that groups together points that are closely packed together (points with many nearby neighbors), marking as outliers points that lie alone in low-density regions.
    * **Performance:** With default or initially chosen parameters (`eps=5`, `min_samples=3`), DBSCAN largely failed to form meaningful clusters. Most points were either assigned to a single large cluster or labeled as noise, indicating that the chosen parameters were not suitable for the density characteristics of this specific dataset. DBSCAN is highly sensitive to its `eps` and `min_samples` parameters and requires careful tuning.

## Evaluation
| Algorithm | Silhouette Score | Davies-Bouldin Index |
| :-------- | :--------------- | :------------------- |
| K-Means   | 0.1749          | 1.6298               |
| DBSCAN    | -0.1953     | 1.7098       |


**Discussion of Results:**
K-Means successfully identified 4 clusters with good compactness and separation, as evidenced by its positive Silhouette Score and reasonable Davies-Bouldin Index. In contrast, DBSCAN, with its current parameters, did not form meaningful clusters, highlighting its sensitivity to parameter tuning and the non-uniform density of the dataset. This suggests K-Means was more robust for this specific dataset and feature representation.

## How to Run
1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Download the dataset:**
    The dataset is referenced from Google Drive. Ensure you have the `unlabelled_sounds` folder located at `/content/drive/MyDrive/Colab_Notebooks/data/unlabelled_sounds` in your Google Drive, or modify the `unlabelled_data_path` variable in the notebook to point to your dataset location.
3.  **Open in Google Colab:** Upload `denys_ntwaritaganzwa's_clustering_assignment.ipynb` to Google Colab.
4.  **Run all cells:** Execute all cells in the notebook sequentially. Ensure your Google Drive is mounted when prompted.

## Dependencies
The notebook requires the following Python libraries:
- `numpy`
- `librosa`
- `matplotlib`
- `seaborn`
- `pandas`
- `scikit-learn` (sklearn)
These can typically be installed via pip:
```bash
pip install numpy librosa matplotlib seaborn pandas scikit-learn
