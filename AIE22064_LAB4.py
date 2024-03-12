import numpy as np
import matplotlib.pyplot as plt

# Load data from file
data = np.loadtxt("kinematic_features.txt")

# Split data into classes
ctrl_data = data[:41]
pd_data = data[41:]

# Calculate mean of both classes
ctrl_class_mean = np.mean(ctrl_data, axis=0)
pd_class_mean = np.mean(pd_data, axis=0)
print(f"Mean for CTRL class: {ctrl_class_mean}")
print(f"Mean for PD class: {pd_class_mean}")

# Calculate standard deviation for each class
ctrl_class_std = np.std(ctrl_data, axis=0)
pd_class_std = np.std(pd_data, axis=0)
print(f"Standard deviation for PD class: {pd_class_std}")
print(f"Standard deviation for CTRL class: {ctrl_class_std}")

# Calculate Euclidean distance between the mean vectors
mean_distance = np.linalg.norm(pd_class_mean - ctrl_class_mean)
print(f"Euclidean distance between means: {mean_distance}")

# Plot histogram for the first feature (velocity)
feature_data = data[:, 0]
plt.figure(figsize=(8, 6))
plt.hist(feature_data, bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of Feature 1')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Calculate mean and variance for the first feature
feature_mean = np.mean(feature_data)
feature_variance = np.var(feature_data)
print(f"Mean of Feature 1: {feature_mean}")
print(f"Variance of Feature 1: {feature_variance}")

# Calculate the Minkowski distance between 2 feature vectors
feature_vector_1 = data[0]  # Selecting the first feature vector
feature_vector_2 = data[43]  # Selecting the second feature vector
minkowski_distances = []
r_values = list(range(1, 11))
for r in r_values:
    distance = np.linalg.norm(feature_vector_1 - feature_vector_2, ord=r)
    minkowski_distances.append(distance)

# Plotting Minkowski Distance against 'r'
plt.figure(figsize=(10, 6))
plt.plot(r_values, minkowski_distances, marker='o', linestyle='-')
plt.title('Minkowski Distance between Two Feature Vectors')
plt.xlabel('r')
plt.ylabel('Distance')
plt.grid(True)
plt.xticks(r_values)
plt.show()
