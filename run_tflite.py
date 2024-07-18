# import matplotlib.pyplot as plt
# import pandas as pd
# from scipy import signal
# import math
# import scipy.io
# from tensorflow.keras.models import Model

import numpy as np
import tensorflow as tf
import os

# Get the current working directory
current_path = os.getcwd()

# Define file names 
trained_model_name = "ms_autoencoder_float16.tflite"
noisy_data_name = "x_test_noisy1.npy"
clean_data_name = "x_test_clean1.npy"

# Full paths to the files 
full_path_model = os.path.join(current_path, trained_model_name)
full_path_noisy = os.path.join(current_path, noisy_data_name)
full_path_clean = os.path.join(current_path, clean_data_name)

# Load the data 
x_test_noisy = np.load(full_path_noisy)
x_test_clean = np.load(full_path_clean)

# Load the tflite model and allocate tensors  
interpreter = tf.lite.Interpreter(model_path=full_path_model)
interpreter.allocate_tensors()
    
# Get input and output tensors 
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
    
# Prepare the test dataset
test_data = x_test_noisy.astype(np.float32)
        
# Run inference on each test sample
results = []

for sample in test_data:
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], sample.reshape((1, 800)))
    
    # Ensure the sample has the correct shape (1, 800)
    # sample = sample.reshape((1, 800))
    
    # Run inference
    interpreter.invoke()
    
    # Get the output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    results.append(output_data)
    
# Convert the results to a NumPy array
results = np.array(results)
results = np.squeeze(results, axis=(1,3))
    
decoded_layer = results
print (decoded_layer)