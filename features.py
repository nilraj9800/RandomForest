import numpy as np
from scipy.spatial import cKDTree as KDTree
from tqdm import tqdm
import pandas as pd


class LidarFeatures:

    def normalize_values(self, values, min_range, max_range):
        min_val = np.nanmin(values)
        max_val = np.nanmax(values)
        normalized_values = (
                (values - min_val) / (max_val - min_val) * (max_range - min_range) + min_range
        )
        return normalized_values

    def calculate_vertical(self, lidar_data, neighborhood_radius):
        vertical_values = np.zeros(len(lidar_data))
        valid_indices = []

        # Create an octree using KDTree
        tree = KDTree(lidar_data)

        for point_idx, point in tqdm(
                enumerate(lidar_data), total=len(lidar_data), desc="Computing verticality"
        ):
            # Query the octree to find neighbors within the specified radius
            neighbor_indices = tree.query_ball_point(point, neighborhood_radius)
            neighbors = lidar_data[neighbor_indices]

            if neighbors.shape[0] >= 3:
                cov_matrix = np.cov(neighbors, rowvar=False)
                eigenvalues, _ = np.linalg.eig(cov_matrix)

                vertical = 1 - (
                        eigenvalues[0] / (eigenvalues[0] + eigenvalues[1] + eigenvalues[2])
                )
                vertical_values[point_idx] = vertical
                valid_indices.append(point_idx)
            else:
                vertical_values[point_idx] = np.nan

        return vertical_values, valid_indices

    def calculate_planar(self, lidar_data, neighborhood_radius):
        planar_values = np.zeros(len(lidar_data))
        tree = KDTree(lidar_data)

        for point_idx, point in tqdm(
                enumerate(lidar_data), total=len(lidar_data), desc="Computing planar"
         ):
            neighbor_indices = tree.query_ball_point(point, neighborhood_radius)
            neighbors = lidar_data[neighbor_indices]

            if neighbors.shape[0] >= 3:
                cov_matrix = np.cov(neighbors, rowvar=False)
                eigenvalues, _ = np.linalg.eig(cov_matrix)
                sorted_eigenvalues = np.sort(eigenvalues)

                planar = (sorted_eigenvalues[1] - sorted_eigenvalues[0]) / (
                        sorted_eigenvalues[0]
                        + sorted_eigenvalues[1]
                        + sorted_eigenvalues[2]
                )
                planar_values[point_idx] = planar
            else:
                planar_values[point_idx] = np.nan

        return planar_values

    def calculate_sphere(self, lidar_data, neighborhood_radius):
        sphere_values = np.zeros(len(lidar_data))

        # Create an octree using KDTree
        tree = KDTree(lidar_data)

        for point_idx, point in tqdm(
                enumerate(lidar_data), total=len(lidar_data), desc="Computing sphere"
        ):
            neighbor_indices = tree.query_ball_point(point, neighborhood_radius)
            neighbors = lidar_data[neighbor_indices]

            if neighbors.shape[0] >= 3:
                cov_matrix = np.cov(neighbors, rowvar=False)
                eigenvalues, _ = np.linalg.eig(cov_matrix)
                sorted_eigenvalues = np.sort(eigenvalues)

                sphere = sorted_eigenvalues[0] / (
                        sorted_eigenvalues[0]
                        + sorted_eigenvalues[1]
                        + sorted_eigenvalues[2]
                )
                sphere_values[point_idx] = sphere
            else:
                sphere_values[point_idx] = np.nan

        return sphere_values

    def calculate_anisotropy(self, lidar_data, neighborhood_radius):
        anisotropy_values = np.zeros(len(lidar_data))

        # Create an octree using KDTree
        tree = KDTree(lidar_data)

        for point_idx, point in tqdm(
                enumerate(lidar_data), total=len(lidar_data), desc="Computing anisotropy"
        ):
            neighbor_indices = tree.query_ball_point(point, neighborhood_radius)
            neighbors = lidar_data[neighbor_indices]

            if neighbors.shape[0] >= 3:
                cov_matrix = np.cov(neighbors, rowvar=False)
                eigenvalues, _ = np.linalg.eig(cov_matrix)
                sorted_eigenvalues = np.sort(eigenvalues)

                anisotropy = (sorted_eigenvalues[2] - sorted_eigenvalues[0]) / (
                        sorted_eigenvalues[0]
                        + sorted_eigenvalues[1]
                        + sorted_eigenvalues[2]
                )
                anisotropy_values[point_idx] = anisotropy
            else:
                anisotropy_values[point_idx] = np.nan

        return anisotropy_values

    def calculate_smallest_eigenvalue(self,lidar_data, neighborhood_radius):
        first_eigenvalue_values = np.zeros(len(lidar_data))

        # Create an octree using KDTree
        tree = KDTree(lidar_data)

        for point_idx, point in tqdm(
                enumerate(lidar_data),
                total=len(lidar_data),
                desc="Computing first eigenvalue",
        ):
            neighbor_indices = tree.query_ball_point(point, neighborhood_radius)
            neighbors = lidar_data[neighbor_indices]

            if neighbors.shape[0] >= 3:
                cov_matrix = np.cov(neighbors, rowvar=False)
                eigenvalues, _ = np.linalg.eig(cov_matrix)
                sorted_eigenvalues = np.sort(eigenvalues)
                first_eigenvalue = sorted_eigenvalues[0]
                first_eigenvalue_values[point_idx] = first_eigenvalue
            else:
                first_eigenvalue_values[point_idx] = np.nan

        return first_eigenvalue_values

    def calculate_surface_variation(self,lidar_data, neighborhood_radius):
        surface_variation_values = np.zeros(len(lidar_data))

        # Create an octree using KDTree
        tree = KDTree(lidar_data)

        for point_idx, point in tqdm(
                enumerate(lidar_data),
                total=len(lidar_data),
                desc="Computing surface variation",
        ):
            neighbor_indices = tree.query_ball_point(point, neighborhood_radius)
            neighbors = lidar_data[neighbor_indices]

            if neighbors.shape[0] >= 3:
                cov_matrix = np.cov(neighbors, rowvar=False)
                eigenvalues, _ = np.linalg.eig(cov_matrix)
                sorted_eigenvalues = np.sort(eigenvalues)
                first_eigenvalue = sorted_eigenvalues[0]
                eigenvalue_sum = np.sum(eigenvalues)
                surface_variation = first_eigenvalue / eigenvalue_sum
                surface_variation_values[point_idx] = surface_variation
            else:
                surface_variation_values[point_idx] = np.nan

        return surface_variation_values

    def calculate_volume_density(self,lidar_points, neighborhood_radius):
        tree = KDTree(lidar_points)
        volume_densities = np.zeros(len(lidar_points))

        for point_idx, point in tqdm(
                enumerate(lidar_points),
                total=len(lidar_points),
                desc="Computing volume density",
        ):
            neighbor_indices = tree.query_ball_point(point, neighborhood_radius)
            neighbors = lidar_points[neighbor_indices]

            # Calculate the volume density
            volume_density = len(neighbors) / (4 / 3 * np.pi * neighborhood_radius ** 3)
            volume_densities[point_idx] = volume_density

        return volume_densities

    def calculate_elevation(self,lidar_points):
        min_z = np.min(lidar_points[:, 2])
        elevation = lidar_points[:, 2] - min_z
        return elevation

    def create_feature_dataframe(self, features, classification_codes, feature_names):
        # Combine all feature arrays into a single 2D NumPy array
        combined_features = np.column_stack(features)
        # Create a Pandas DataFrame from the combined feature array
        feature_df = pd.DataFrame(combined_features, columns=feature_names)
        # Add the classification codes as a new column in the DataFrame
        feature_df['classification'] = classification_codes
        return feature_df

    def predict_feature(self, lidar_point, neighborhood_radius):
        # Calculate the vertical of all points with a neighborhood radius of 5

        new_feature1, valid_point_indices = self.calculate_vertical(lidar_point, neighborhood_radius)

        new_feature2 = self.calculate_planar(lidar_point, neighborhood_radius)
        new_feature3 = self.calculate_sphere(lidar_point, neighborhood_radius)
        new_feature4 = self.calculate_anisotropy(lidar_point, neighborhood_radius)

        new_feature5 = self.calculate_smallest_eigenvalue(lidar_point, neighborhood_radius)

        new_feature6 = self.calculate_surface_variation(lidar_point, neighborhood_radius)

        new_feature7 = self.calculate_elevation(lidar_point)

        return (new_feature1, new_feature2, new_feature3, new_feature4,
                new_feature5, new_feature6, new_feature7)
