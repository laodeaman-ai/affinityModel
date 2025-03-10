# Predicting Binding Affinity of PPAR Inhibitors with Deep Learning

📘 **Description:**
This repository contains a Python script for predicting the binding affinity of PPAR inhibitors based on their Morgan fingerprints. It leverages a deep learning model built with TensorFlow and visualizes the prediction results with regression plots. The script handles data preprocessing, model training, evaluation, and result visualization.

🚀 **Features:**
- **Data Preprocessing:**
  - Load molecular fingerprints from `ppar_morgan.csv`.
  - Filter molecules with binding affinity ≤ -5.
  - Normalize binding affinity values.
  
- **Model Training:**
  - Fully connected neural network with ReLU activations.
  - Configurable number of epochs and batch size.
  - Train/test split with automatic sample count logging.

- **Evaluation:**
  - Mean Squared Error (MSE) and Mean Absolute Error (MAE).
  - R² scores for both training and test sets.

- **Visualization:**
  - Scatter plot comparing experimental vs predicted binding affinity.
  - Linear regression lines for training and test sets.
  - Save plot as `ppar_morgan_plot.png`.

🔧 **Requirements:**
- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- TensorFlow
- Matplotlib

📂 **File Structure:**
```
├── ppar_morgan.csv              # Input CSV with fingerprints & binding affinity
├── ppar_affinity_model.py       # Main script for training & evaluation
├── ppar_morgan_model.keras      # Saved trained model
├── ppar_morgan_plot.png         # Plot showing experimental vs predicted affinity
```

⚙️ **Usage:**
1. Install the required packages:
```bash
pip install pandas numpy scikit-learn tensorflow matplotlib
```

2. Run the script:
```bash
python ppar_affinity_model.py
```

3. Check the output metrics and visualization.

🔍 **Example:**
**Metrics:**
```
Number of molecules in Training Set: 320
Number of molecules in Test Set: 80
Mean Squared Error: 0.05
Mean Absolute Error: 0.18
R-squared (Training Set): 0.92
R-squared (Test Set): 0.87
```

**Plot:**
- Blue points: Training set predictions
- Green points: Test set predictions
- Solid lines: Linear regression fits

📘 **Customization:**
- Adjust the model architecture:
```python
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
```

- Change training parameters:
```python
model.fit(X_train, y_train, epochs=50, batch_size=32)
```

📩 **Contributing:**
Feel free to fork, submit issues, or suggest improvements via pull requests!

🔖 **License:**
MIT License.

---

Happy modeling and drug discovery! 🚀🧬

---

Let me know if you want any adjustments or extra details! ✌️

