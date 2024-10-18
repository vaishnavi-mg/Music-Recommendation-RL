# Music Playlist Generator Using Clustering and Reinforcement Learning

## Overview

This project aims to generate personalized music playlists based on song features using **Clustering Algorithms** and **Reinforcement Learning (RL)** methods like Q-Learning and Policy Gradient. The dataset contains various audio features of songs, and the goal is to cluster songs into distinct mood groups and use RL to generate playlists that match a user’s preferred mood.

## Dataset

The dataset contains several **audio features** of songs that help in determining their mood and style. These features are used to cluster songs into three distinct mood groups: **Energetic**, **Chill**, and **Acoustic**.

### Audio Features:
- **`energy_%`**: Represents the energy level of the song (higher values indicate more energetic tracks).
- **`danceability_%`**: Measures how suitable the song is for dancing (higher values indicate better danceability).
- **`acousticness_%`**: Reflects the acoustic quality of the song (higher values suggest more acoustic elements).
- **`track_name`**: The title of the song.
- **`artist(s)_name`**: The name(s) of the artist(s) performing the song.

## Clustering Process

To group songs into mood clusters, **K-Means Clustering** is used. The features `energy_%`, `danceability_%`, and `acousticness_%` are selected for clustering. The algorithm groups songs into three distinct clusters based on their audio characteristics:

1. **Cluster 0: Energetic** - High energy, suitable for upbeat moods.
2. **Cluster 1: Chill** - Calm and relaxing, ideal for laid-back moods.
3. **Cluster 2: Acoustic** - Acoustic-heavy songs with lower energy.

### Clustering Algorithm: K-Means
- The features are **standardized** to ensure proper clustering.
- **3 clusters** are defined for the moods: Energetic, Chill, and Acoustic.
- After clustering, the dataset is labeled with the respective cluster for each song.

## Reinforcement Learning Process

Two **Reinforcement Learning** algorithms are implemented to generate personalized playlists for a given mood cluster:

1. **Q-Learning Playlist Generator**: 
   - Learns through feedback and rewards (simulated likes/dislikes) to recommend songs from the desired mood cluster.
   - It iteratively improves song selection based on the user’s feedback.

2. **Policy Gradient Playlist Generator**:
   - Selects songs based on a probability-based policy, where songs from the desired mood cluster are assigned probabilities, and a playlist is generated based on these probabilities.
   - The algorithm emphasizes exploration to select different songs.

## Usage

- **Clustering**: The dataset is clustered using K-Means, grouping songs into three clusters (Energetic, Chill, Acoustic).
- **Reinforcement Learning**: Playlists are generated using Q-Learning and Policy Gradient methods, allowing you to explore which RL method provides better recommendations.

## Visualizations and Evaluation

- **Silhouette Score**: Evaluates the quality of clustering.
- **Dendrogram**: Visualizes hierarchical relationships between songs based on their features.
- **Scatter Plot**: Shows the distribution of clustered songs.

## How to Run

1. Load the dataset (update the path in the script).
2. Execute the K-Means clustering process to group the songs.
3. Use Q-Learning or Policy Gradient algorithms to generate playlists based on the mood cluster you choose.
4. Visualize the clusters and evaluate playlist generation using accuracy metrics.

## Results

After training, the RL agents will generate personalized playlists based on your chosen mood. You can compare the performance of Q-Learning and Policy Gradient methods and visualize the clustering results.






