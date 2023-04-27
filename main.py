# main.py

import laspy
import numpy as np
from features import LidarFeatures
from sklearn.impute import SimpleImputer
from lidar_classification import LidarClassification
from sklearn.model_selection import train_test_split


def main():
    # Load LiDAR data (assuming LAS file format)
    las_file = 'M:\\lidar\\Test\\building.las'
    lidar_data = laspy.read(las_file)
    lidar_points = np.vstack((lidar_data.x, lidar_data.y, lidar_data.z)).T
    neighborhood_radius = 3

    # Create an instance of LidarFeatures
    feature_calculator = LidarFeatures()

    # ... Calculate features using the LidarFeatures class ...
    vertical_values, valid_point_indices = feature_calculator.calculate_vertical(lidar_points, neighborhood_radius)
    planar_values = feature_calculator.calculate_planar(lidar_points, neighborhood_radius)
    sphere_values = feature_calculator.calculate_sphere(lidar_points, neighborhood_radius)
    anisotropy_values = feature_calculator.calculate_anisotropy(lidar_points, neighborhood_radius)
    smallest_eigenvalue = feature_calculator.calculate_smallest_eigenvalue(lidar_points, neighborhood_radius)
    surface_variation_values = feature_calculator.calculate_surface_variation(lidar_points, neighborhood_radius)
    elevations = feature_calculator.calculate_elevation(lidar_points)

    features = [vertical_values, planar_values, sphere_values, anisotropy_values, smallest_eigenvalue,
                surface_variation_values, elevations]
    # List of feature names corresponding to the feature arrays
    feature_names = ['vertical', 'planar', 'sphere', 'anisotropy', 'smallest_eigen', 'surfaceVariation',
                     'elevations']
    # Get classification codes from the LAS data
    classification_codes = lidar_data.classification
    # Create the DataFrame
    feature_dataframe = feature_calculator.create_feature_dataframe(features, classification_codes, feature_names)
    # Remove rows with missing values (NaN)
    feature_dataframe = feature_dataframe.dropna()

    # Split the data into training and testing sets
    x = feature_dataframe.drop('classification', axis=1)
    y = feature_dataframe['classification']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Create an instance of LidarClassification
    classifier = LidarClassification(x_train, y_train)

    # Train the Random Forest classifier
    classifier.train_classifier(100, 4, 42)

    # ... Evaluate the classifier and predict class labels for new data ...

    # classifier.evaluate_classifier(x_test, y_test)

    new_data = 'M:\\lidar\\las\\building.las'
    las = laspy.read(new_data)
    lidar_point = np.vstack((las.x, las.y, las.z)).T
    new_feature1, new_feature2, new_feature3, new_feature4, new_feature5, new_feature6, new_feature7 = \
        feature_calculator.predict_feature(lidar_point, neighborhood_radius)

    feature_matrix = np.column_stack((new_feature1, new_feature2, new_feature3,
                                      new_feature4, new_feature5, new_feature6, new_feature7))
    # Replace infinity values with NaN
    feature_matrix = np.where(np.isinf(feature_matrix), np.nan, feature_matrix)
    # Impute missing values using the mean of the corresponding feature column
    impute = SimpleImputer(strategy='mean')
    feature_matrix_clean = impute.fit_transform(feature_matrix)
    predicted_label = classifier.predict_class_labels(feature_matrix_clean)
    las.classification = predicted_label.astype(np.uint8)
    output_las_file = 'M:\\lidar\\RandomForest\\building_classified20.las'
    las.write(output_las_file)


if __name__ == "__main__":
    main()

