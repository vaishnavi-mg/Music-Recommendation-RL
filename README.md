# Music-Recommendation-RL
Music playlist generator using Reinforcement Learning (QLearning &amp; Policy Gradient). This project clusters songs based on mood (Energetic, Chill, Acoustic) and compares QLearning and Policy Gradient algorithms for playlist generation. 

The dataset contains several audio features of songs that help in determining their mood and style. These features are used to cluster songs into three distinct mood groups: Energetic, Chill, and Acoustic.

Audio Features:
energy_%: Represents the energy level of the song (higher values indicate more energetic tracks).
danceability_%: Measures how suitable the song is for dancing (higher values indicate better danceability).
acousticness_%: Reflects the acoustic quality of the song (higher values suggest more acoustic elements).
track_name: The title of the song.
artist(s)_name: The name(s) of the artist(s) performing the song.
These features are analyzed and processed to form three main mood clusters, which are then used as the basis for recommending personalized playlists.



