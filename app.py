import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
import numpy as np
import random

# Load the dataset (Update the file path if necessary)
file_path = '/content/Book1.xlsx'  # Replace this with your file path
data = pd.read_excel(file_path)

# Step 1: Inspect dataset columns
print("Columns in the dataset:\n", data.columns)

# Step 2: Choose relevant features based on your dataset
# Update the feature names with correct column names from your data
features = ['energy_%', 'danceability_%', 'acousticness_%']  # Updated with correct column names
print("Selected Features: ", features)

# Step 3: Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[features])

# Step 4: KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # You can change the number of clusters
kmeans.fit(scaled_data)

# Initial clusters
initial_clusters = kmeans.labels_
print("Initial Clusters:\n", initial_clusters)

# Step 5: Find final clusters and error rate
epochs = kmeans.n_iter_
final_clusters = kmeans.labels_
error_rate = kmeans.inertia_  # Sum of squared distances to the closest centroid

print(f"Final Clusters after {epochs} epochs:\n", final_clusters)
print(f"Error Rate: {error_rate}")

# Step 6: Visualizing KMeans Clustering
plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=final_clusters, cmap='viridis')
plt.title("KMeans Clustering Visualization")
plt.xlabel(features[0])
plt.ylabel(features[1])
plt.show()

# Step 7: Hierarchical Clustering
Z = linkage(scaled_data, method='ward')

# Plot the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(Z, truncate_mode='level', p=3, labels=data.index)
plt.title("Hierarchical Clustering Dendrogram")
plt.show()

# Silhouette Score to check clustering quality
sil_score = silhouette_score(scaled_data, final_clusters)
print(f"Silhouette Score: {sil_score}")

# Define moods for clusters
cluster_moods = {0: 'Energetic', 1: 'Chill', 2: 'Acoustic'}

# QLearningPlaylist class definition
class QLearningPlaylist:
    def __init__(self, data, cluster, n_songs=15):
        self.data = data
        self.cluster = cluster
        self.n_songs = n_songs
        self.q_table = {}  # Initialize Q-table for state-action pairs

    def initialize_q_table(self):
        """Initialize Q-table for each song with random Q-values."""
        for song_id in self.data.index:
            self.q_table[song_id] = random.uniform(0, 1)  # Random Q-value initialization

    def update_q_value(self, song_id, reward, learning_rate=0.1, discount_factor=0.95):
        """Update Q-values using the Q-learning formula."""
        old_q_value = self.q_table[song_id]
        max_next_q = max(self.q_table.values())  # Max future reward
        # Q-learning update rule
        self.q_table[song_id] = old_q_value + learning_rate * (reward + discount_factor * max_next_q - old_q_value)

    def generate_playlist(self):
        """Generate a playlist of top songs using Q-learning."""
        self.initialize_q_table()
        playlist_songs = []
        
        for _ in range(self.n_songs):
            # Select song with max Q-value in the cluster
            cluster_songs = self.data[final_clusters == self.cluster].index
            best_song_id = max(cluster_songs, key=lambda x: self.q_table[x])
            playlist_songs.append(best_song_id)
            
            # Simulate user feedback (like = +1 reward, dislike = -1 reward)
            feedback = random.choice([1, -1])  # Simulated feedback
            self.update_q_value(best_song_id, reward=feedback)
        
        playlist = self.data.loc[playlist_songs][['track_name', 'artist(s)_name']]
        return playlist

# Policy Gradient Algorithm (Simplified version)
class PolicyGradientPlaylist:
    def __init__(self, data, cluster, n_songs=15):
        self.data = data
        self.cluster = cluster
        self.n_songs = n_songs

    def generate_playlist(self):
        """Generate a playlist using a probability-based policy."""
        cluster_songs = self.data[final_clusters == self.cluster]
        # Assign random probabilities to each song in the cluster
        probabilities = np.random.dirichlet(np.ones(len(cluster_songs)), size=1)[0]
        song_ids = cluster_songs.index
        
        # Select songs based on probability distribution
        selected_songs = np.random.choice(song_ids, size=self.n_songs, replace=False, p=probabilities)
        playlist = self.data.loc[selected_songs][['track_name', 'artist(s)_name']]
        
        return playlist

# Example usage:
user_input_cluster = int(input("Enter cluster number (0 - ENERGETIC , 1-CHILL AND 2-ACOUSTIC) to generate playlist: "))
mood = cluster_moods.get(user_input_cluster, "Unknown Mood")
print(f"Generating playlist for mood: {mood}")

# Q-learning based playlist generation
q_learning = QLearningPlaylist(data, user_input_cluster)
q_learning_playlist = q_learning.generate_playlist()
print(f"\nGenerated Playlist using Q-learning for Cluster {mood}:\n", q_learning_playlist)

# Policy Gradient based playlist generation
policy_gradient = PolicyGradientPlaylist(data, user_input_cluster)
policy_gradient_playlist = policy_gradient.generate_playlist()
print(f"\nGenerated Playlist using Policy Gradient for Cluster {mood}:\n", policy_gradient_playlist)
