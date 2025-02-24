import os
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.merge import merge
from rasterio.mask import mask
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

class CompositeImageClassifier:
    def __init__(self, base_folder, shapefile, n_estimators=100, random_state=60):
        self.base_folder = base_folder  # Main dataset directory
        self.shapefile = shapefile  # Shapefile for feature extraction
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state)
        self.data, self.labels = [], []

    def merge_bands(self, band_paths):
        """Merge multiple bands into a single composite image."""
        bands = [rasterio.open(band).read(1) for band in band_paths]
        return np.stack(bands, axis=0)  # Stack bands along depth axis

    def merge_tiles(self, tile_folder):
        """Merge multiple tile composites into one image while keeping spatial reference."""
        src_files = []
        for tile in os.listdir(tile_folder):
            tile_path = os.path.join(tile_folder, tile)
            if os.path.isdir(tile_path):
                band_paths = [os.path.join(tile_path, f) for f in os.listdir(tile_path)]
                composite = self.merge_bands(band_paths)
                src = rasterio.open(band_paths[0])  # Use first band as reference
                src_files.append((composite, src.transform))
        
        if src_files:
            merged, transform = merge([r[0] for r in src_files], [r[1] for r in src_files])
            return merged, transform
        return None, None

    def extract_feature_extent(self, composite_image, transform):
        """Extract the region of interest using the shapefile."""
        gdf = gpd.read_file(self.shapefile)
        with rasterio.open(self.shapefile.replace('.shp', '.tif'), 'w', driver='GTiff',
                           height=composite_image.shape[1], width=composite_image.shape[2],
                           count=composite_image.shape[0], dtype=composite_image.dtype,
                           transform=transform) as dst:
            dst.write(composite_image)
        
        with rasterio.open(self.shapefile.replace('.shp', '.tif')) as src:
            out_image, out_transform = mask(src, gdf.geometry, crop=True)
        return out_image

    def process_yearly_data(self):
        """Process all years' data, merging tiles and extracting features."""
        for year in os.listdir(self.base_folder):
            year_path = os.path.join(self.base_folder, year)
            if os.path.isdir(year_path):
                composite_image, transform = self.merge_tiles(year_path)
                if composite_image is not None:
                    feature_extent = self.extract_feature_extent(composite_image, transform)
                    self.extract_features(feature_extent, year)
        
    def extract_features(self, composite_image, year):
        """Extract feature vectors from the composite image."""
        height, width, bands = composite_image.shape
        for i in range(height):
            for j in range(width):
                self.data.append(composite_image[:, i, j].flatten())  # Flatten pixel bands
                self.labels.append(year)  # Assign label (year)
        
    def train_model(self):
        """Train Random Forest model."""
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=0.2, random_state=self.random_state)
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        print(f"Accuracy: {accuracy_score(y_test, predictions)}")
        print(f"F1 Score: {f1_score(y_test, predictions, average='weighted')}")
        
    def predict(self, new_composite_image):
        """Predict using trained model."""
        flattened_features = new_composite_image.reshape(-1, new_composite_image.shape[0])
        return self.model.predict(flattened_features)

# Example usage:
base_folder = "path_to_dataset"
shapefile = "path_to_shapefile.shp"
classifier = CompositeImageClassifier(base_folder, shapefile)
classifier.process_yearly_data()
classifier.train_model()

# MIT License
# Copyright (c) 2025 Adewuyi Adewale Isaiah
# See LICENSE file for more details.

