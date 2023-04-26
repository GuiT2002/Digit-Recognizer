import tensorflow as tf
import numpy as np
import pandas as pd

# File path of the test.csv file
file_path = "\\path_to_file\\test.csv"

# Read the test.csv file into a DataFrame
test_df = pd.read_csv(file_path)

# Extract test data as numpy array
test_data = test_df.values

# Reshape test data to (num_samples, 28, 28, 1)
test_data = test_data.reshape(-1, 28, 28, 1)

# Normalize test data
test_data = test_data / 255.0

# Load the trained model
model = tf.keras.models.load_model('DR_v6')

# Make predictions on test data
predictions = model.predict(test_data)

# Get predicted labels
predicted_labels = np.argmax(predictions, axis=1)

# Convert predicted labels to strings
predicted_labels_str = predicted_labels.astype(str)

# Create a DataFrame to store results
results_df = pd.DataFrame({'ImageId': range(1, len(predicted_labels) + 1),
                          'Label': predicted_labels_str})

# Write the DataFrame to a CSV file
results_df.to_csv('predictions5.csv', index=False)
