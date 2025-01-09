import tensorflow as tf
import pandas as pd
import numpy as np
import os  # Import os module to check file existence
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import ast
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping  # Import EarlyStopping


def formatkps(row):
    # Extract keypoint values from the row
    keypoints = []
    valid = 0
    for i in range(1, 5):  # For Person1 to Person4
        for part in ['head', 'neck', 'left_ear', 'right_ear', 'left_shoulder', 'left_elbow', 
                     'left_hand', 'right_shoulder', 'right_elbow', 'right_hand']:
            # Construct the column name
            col_name = f'Person{i}_{part}'
            try:
                x, y = ast.literal_eval(row[col_name])
                valid = valid+1
            except:
                x, y = -1, -1

            # Append x and y values separately to the keypoints list
            keypoints.append(x)
            keypoints.append(y)
    # print("Valid:",valid)
    return keypoints

# Define the hybrid model
def build_hybrid_model(input_image_shape, input_keypoints_dim):
    # Image input and feature extraction
    image_input = layers.Input(shape=input_image_shape, name="image_input")
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, weights='imagenet', input_tensor=image_input
    )
    base_model.trainable = False  # Freeze pre-trained model weights
    image_features = layers.GlobalAveragePooling2D()(base_model.output)

    # Keypoints input and processing
    keypoints_input = layers.Input(shape=(input_keypoints_dim,), name="keypoints_input")
    keypoints_features = layers.Dense(64, activation='relu')(keypoints_input)
    # keypoints_features = layers.Dropout(0.3)(keypoints_features)

    # Concatenate image and keypoint features
    combined_features = layers.concatenate([image_features, keypoints_features])

    # Fully connected layers for classification
    x = layers.Dense(256, activation='relu')(combined_features)
    x = layers.Dense(128, activation='relu')(combined_features)
    # x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(4, activation='sigmoid', name="output")(x)  # Sigmoid activation for binary classification

    # Build and compile model
    model = models.Model(inputs={"image_input": image_input, "keypoints_input": keypoints_input}, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Use binary crossentropy
    return model

from sklearn.preprocessing import LabelEncoder
import os

from sklearn.preprocessing import LabelEncoder
import os
import numpy as np
import pandas as pd

# Custom data generator
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, images_folder, batch_size, target_image_size=(1280, 720)):
        self.data = data
        self.images_folder = images_folder
        self.batch_size = batch_size
        self.target_image_size = target_image_size
        self.indices = np.arange(len(self.data))
        
        # Initialize LabelEncoder
        self.label_encoder = LabelEncoder()
        # Fit the label encoder for hand raised labels (0 or 1)
        self.label_encoder.fit([0, 1])  # Encoding binary values (0, 1)
    
    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        images = []
        keypoints = []
        labels = []
        
        for i in batch_indices:
            row = self.data.iloc[i]
            # Construct image file path
            image_path = f"{self.images_folder}/frame_{int(row['frame']):04d}.jpg"
            
            # Skip the row if the image file does not exist
            if not os.path.exists(image_path):
                continue
            
            # Load and preprocess image
            image = tf.keras.utils.load_img(image_path, target_size=self.target_image_size)
            image = tf.keras.utils.img_to_array(image) / 255.0  # Normalize to [0, 1]
            images.append(image)
            
            # Load keypoints
            mykeypoints = formatkps(row)  # Process keypoints
            keypoints.append(mykeypoints)
            
            # Modify labels: binary vector for each person (1 if hand raised, 0 if not)
            encoded_labels = [
                self._get_label(row, 'Person1_hand_raised'),
                self._get_label(row, 'Person2_hand_raised'),
                self._get_label(row, 'Person3_hand_raised'),
                self._get_label(row, 'Person4_hand_raised')
            ]
            labels.append(encoded_labels)
        
        # If no valid images were found in the batch, skip this batch
        if len(images) == 0:
            return self.__getitem__((index + 1) % self.__len__())  # Recursively get the next batch
        
        images = np.array(images)
        keypoints = np.array(keypoints)
        labels = np.array(labels)
        
        return {"image_input": images, "keypoints_input": keypoints}, labels
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)
    
    def _get_label(self, row, column_name):
        """
        This function checks if the label value is missing (NaN) and replaces it with 0.
        """
        label_value = row[column_name]
        if pd.isna(label_value) or label_value is None:
            return 0  # Replace missing label with 0
        else:
            return int(label_value)  # Ensure the value is either 0 or 1


# Paths
images_folder = "C:/OsamaEjaz/Qiyas_Gaze_Estimation/Wajahat_Yolo_keypoint/hybrid/frmaes/"
csv_file = "C:/OsamaEjaz/Qiyas_Gaze_Estimation/Wajahat_Yolo_keypoint/hybrid/combined.csv"

# Load CSV to get the data
data = pd.read_csv(csv_file)

# Train-validation split
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Batch size
batch_size = 32

# Create data generators
train_generator = DataGenerator(train_data, images_folder, batch_size=batch_size)
val_generator = DataGenerator(val_data, images_folder, batch_size=batch_size)

# Build the model
input_image_shape = (1280, 720, 3)
input_keypoints_dim = 80
model = build_hybrid_model(input_image_shape, input_keypoints_dim)

physical_devices = tf.config.list_physical_devices('GPU')

early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=25,          # Stop after 3 epochs with no improvement
    restore_best_weights=True,  # Restore the model weights from the epoch with the best value of the monitored quantity
    verbose=1
)
# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=200,
    callbacks=[early_stopping]
)

# Save the model
model.save("hybrid_model.h5")
