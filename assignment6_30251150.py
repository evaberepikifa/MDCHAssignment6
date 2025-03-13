import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

#!pip install tensorflow-hub
#load dataset
data = pd.read_csv('processed.cleveland.data', header=None)
data.columns

# Separate features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

#Handle NAn values
X.replace('?', np.nan, inplace=True)

#categorical and continous columns
continuous_cols= X.iloc[:, [0, 3, 4, 7, 9, 11]]
categorical_cols = X.iloc[:, [1, 2, 5, 6, 8, 10, 12]]

# Imputers : mean and mode
imp_cat = SimpleImputer(strategy='most_frequent')
imp_cont = SimpleImputer(strategy='mean')
#impute missing values
X_cont_imputed = pd.DataFrame(imp_cont.fit_transform(continuous_cols), columns=continuous_cols.columns)
X_cat_imputed = pd.DataFrame(imp_cat.fit_transform(categorical_cols), columns=categorical_cols.columns)

#one hot encode categorical variable    
X_cat_imputed.columns = X_cat_imputed.columns.astype(str)
# One-hot encode categorical variable
encoder = OneHotEncoder(drop='first', sparse_output=False)
ohe_cat = encoder.fit_transform(X_cat_imputed)
ohe_cat = pd.DataFrame(ohe_cat, columns=encoder.get_feature_names_out(X_cat_imputed.columns))

#scale continuous variable
scaler = StandardScaler()
X_cont_scaled = scaler.fit_transform(X_cont_imputed)

#one hot encode target variable
y_encoded = to_categorical(y,num_classes=5)

#combine into one X
X_combined = np.concatenate([ohe_cat, X_cont_scaled ], axis=1)

#split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_encoded, test_size=0.3, random_state=42)

best_neurons = 0
best_val_accuracy = 0
best_model = None

# Loop through different hidden layer sizes
for neurons in range(5, 55, 5):
    print(f" {neurons} neurons in the hidden layer")
    
    # model
    model = Sequential([
        Dense(neurons, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(5, activation='softmax')  
    ])
    
    # Compilation of model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Training the model
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=25,
        validation_split=0.2,
        verbose=0 
    )
    
    # Print final validation accuracy
    val_accuracy = history.history['val_accuracy'][-1]
    print(f"Final validation accuracy with {neurons} neurons: {val_accuracy:.4f}\n")

    # Find the best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_neurons = neurons
        best_model = model

print(f"Best number of neurons: {best_neurons} with validation accuracy: {best_val_accuracy:.4f}\n")

# Evaluate the best model on the test data
test_loss, test_accuracy = best_model.evaluate(X_test, y_test, verbose=0)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")





