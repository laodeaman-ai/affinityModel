import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv('ppar_morgan.csv')

# Filter rows where binding_affinity <= -5
df_filtered = df[df['binding_affinity'] <= -5]

# Prepare features (Morgan fingerprints) and labels (binding_affinity)
# Convert Morgan fingerprint from bitstring to a numpy array of integers
def fingerprint_to_array(fingerprint):
    return np.array([int(bit) for bit in fingerprint])

# Extract features and labels
X = np.array(df_filtered['morgan_fingerprint'].apply(fingerprint_to_array).tolist())
y = df_filtered['binding_affinity'].values

# Normalize the binding affinity values
scaler = StandardScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()  # Normalize the target (binding affinity)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.2, random_state=42)

# Print the number of samples in training and test sets
num_train_samples = len(X_train)
num_test_samples = len(X_test)
print(f'Number of molecules in Training Set: {num_train_samples}')
print(f'Number of molecules in Test Set: {num_test_samples}')

# Build the deep learning model
model = models.Sequential()
model.add(layers.InputLayer(input_shape=(X_train.shape[1],)))  # Input layer
model.add(layers.Dense(128, activation='relu'))  # First hidden layer
model.add(layers.Dense(64, activation='relu'))   # Second hidden layer
model.add(layers.Dense(64, activation='relu'))   # Third hidden layer
model.add(layers.Dense(1))  # Output layer (regression)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Predict on training and test data
y_pred_train_scaled = model.predict(X_train)
y_pred_test_scaled = model.predict(X_test)

# Inverse transform the predicted values to the original scale
y_pred_train = scaler.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).flatten()
y_pred_test = scaler.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).flatten()

# Inverse transform the true values to the original scale
y_train_original = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Calculate R-squared for both train and test sets
r2_train = r2_score(y_train_original, y_pred_train)
r2_test = r2_score(y_test_original, y_pred_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test_original, y_pred_test)
mae = np.mean(np.abs(y_test_original - y_pred_test))

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared (Training Set): {r2_train}")
print(f"R-squared (Test Set): {r2_test}")

# Optionally, save the trained model
model.save('ppar_morgan_model.keras')

# Create LinearRegression models for the training and test sets to plot regression lines
train_lr = LinearRegression()
test_lr = LinearRegression()

# Train linear models
train_lr.fit(y_train_original.reshape(-1, 1), y_pred_train)
test_lr.fit(y_test_original.reshape(-1, 1), y_pred_test)

# Predict using the linear models
train_line = train_lr.predict(y_train_original.reshape(-1, 1))
test_line = test_lr.predict(y_test_original.reshape(-1, 1))

# Plotting the comparison between true and predicted values (combined for train and test)
plt.figure(figsize=(8, 8))

# Scatter plot for training set and test set
plt.scatter(y_train_original, y_pred_train, color='blue', alpha=0.6, label=f'Training Set ({num_train_samples} samples)', s=40)
plt.scatter(y_test_original, y_pred_test, color='green', alpha=0.6, label=f'Test Set ({num_test_samples} samples)', s=40)

# Add the regression line for training set
plt.plot(y_train_original, train_line, color='blue', linestyle='-', label=f'Training Set Regression (R²: {r2_train:.3f})')

# Add the regression line for test set
plt.plot(y_test_original, test_line, color='green', linestyle='-', label=f'Test Set Regression (R²: {r2_test:.3f})')

# Add labels and title with R-squared values in the title
plt.title(f'Experimental vs Predicted Binding Affinity for PPAR Inhibitors')
plt.xlabel('Experimental Binding Affinity')
plt.ylabel('Predicted Binding Affinity')

# Show legend
plt.legend()

# Save the plot to a file (PNG format)
plt.tight_layout()
plt.savefig('ppar_morgan_plot.png')  # Save as PNG

# Show the plot
plt.show()

