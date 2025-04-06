import tensorflow as tf
from tensorflow import keras

try:
    # Load model without compilation
    old_model = tf.keras.models.load_model('./saved_models/final_model.h5', compile=False)
    print("✅ Model loaded successfully!")

    # Extract input shape and remove batch dimension
    input_shape = tuple(dim for dim in old_model.input_shape[1:])  # Ensuring no batch shape
    print(f"ℹ️ Extracted input shape: {input_shape}")

    # Rebuild the model from scratch (Functional API)
    new_input = keras.layers.Input(shape=input_shape, dtype='float32', name="new_input")
    new_output = old_model(new_input)  # Pass new input through the existing model
    new_model = keras.Model(inputs=new_input, outputs=new_output)

    # Save in both H5 and TensorFlow SavedModel format
    new_model.save('./saved_models/converted_model.h5')
    print("✅ Model successfully saved in H5 format.")

    new_model.save('./saved_models/converted_model', save_format='tf')
    print("✅ Model successfully saved in TensorFlow SavedModel format.")

except Exception as e:
    print(f"❌ Error: {e}")
