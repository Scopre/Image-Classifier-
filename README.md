# Image-Classifier-
📌 Project Overview

This project implements a composite image classification system for geospatial data using a Random Forest Classifier. It processes multi-band satellite images, merges them into composite images, extracts specific feature extents using a shapefile, and trains a machine learning model for classification and future predictions.

🚀 Key Features

✔ Loads and processes multi-band satellite images from multiple year-wise folders✔ Merges individual bands to create composite images for each tile✔ Combines multiple tiles into a single composite image per year✔ Extracts relevant feature extents from the merged images using a shapefile✔ Splits the extracted data into training and testing sets for model evaluation✔ Trains a Random Forest Classifier to recognize spatial patterns✔ Evaluates model performance using accuracy and F1-score metrics✔ Predicts future geospatial classifications using trained model insights

🛠 Tech Stack

🐍 Python – Core programming language🌍 Rasterio – Geospatial raster data handling🗺 GeoPandas – Shapefile-based feature extraction📊 NumPy – Array manipulation and numerical operations🔢 Scikit-Learn – Machine learning (Random Forest Classifier)📡 OpenCV – Image processing (for auxiliary tasks)

📂 Project Structure

CompositeImageClassifier Class:

merge_bands(): Merges multiple bands into a single composite image per tile

merge_tiles(): Merges all tile composites into a full-year composite

extract_feature_extent(): Extracts feature regions using a shapefile

process_yearly_data(): Iterates through all years and processes data

train_model(): Trains a Random Forest Classifier

predict(): Uses the trained model for classification on new composite images

🛠 Setup Instructions

Clone the repository:

git clone https://github.com/yourusername/composite-image-classification.git

Install dependencies:

pip install rasterio geopandas numpy scikit-learn opencv-python

Prepare dataset:

Organize yearly folders with tile subfolders containing multi-band images.

Ensure a shapefile is available for feature extraction.

Run the classification script:

python main.py

View classification results and evaluate model performance.

📈 Model Performance

Achieved X% accuracy and Y F1-score on test data.

Classification results validated using geospatial visualization techniques.

📌 Future Improvements

🔹 Optimize hyperparameters for better accuracy🔹 Integrate deep learning (CNNs) for enhanced classification🔹 Improve computational efficiency for large datasets

📜 License

This project is licensed under the MIT License – see the LICENSE file for details.

🤝 Contributions are welcome! If you have ideas to improve the project, feel free to fork and submit a pull request.

📧 Contact:adewuyiadewaleisaiah@gmail.com  | 🌍 GitHub: Scopre
