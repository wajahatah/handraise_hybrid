import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import LabelEncoder


# Function to load data from folders
def load_data_from_folders(csv, images_folder):
    images = []
    X_x = []
    X_y = []
    labels = []

    def parse_feature(value):
        if isinstance(value, str) and value.startswith("(") and value.endswith(")"):
            # Extract the first number from the tuple-like string
            # x,y = float(value.split(",")[0].strip("()"))
            x,y = value.strip("()").split(",")
            return float(x),float(y)
        else: 
            return float(value), float(0)
        
    data = pd.read_csv(csv)

    feature_columns = [col for col in data.columns if "_head" in col or "_neck" in col or "_ear" in col or "_shoulder" in col or "_elbow" in col or "_hand" in col]
    label_columns = [col for col in data.columns if "_hand_raised" in col]

    # Initialize dataframes for x and y coordinates
    X_x = {col: [] for col in feature_columns}
    X_y = {col: [] for col in feature_columns}

    for feature in feature_columns:
        x, y = zip(*data[feature].map(parse_feature))
        X_x[feature] = x
        X_y[feature] = y

    # Load labels for each person
    for label_col in label_columns:
        encoder = LabelEncoder()
        labels[label_col] = encoder.fit_transform(data[label_col])

    # Load images
    for _, row in data.iterrows():
        frame_number = int(row['frame'])
        img_name = f"frame_{frame_number:04d}.jpg"
        img_path = os.path.join(images_folder, img_name)

        if os.path.exists(img_path):
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # Normalize
            images.append(img_array)
        else:
            print(f"Image {img_name} not found in {images_folder}. Skipping.")

    return images, X_x,X_y, labels


    """    
    X = data.drop(columns=['frame', 'Person1_hand_raised','Person2_hand_raised','Person3_hand_raised','Person4_hand_raised'])
    
    X_x = pd.DataFrame()
    X_y = pd.DataFrame()
    for feature in X:
        X_x[feature], X_y[feature] = zip(*data[feature].map(parse_feature))
    # yt = data['hand_raised']
    yt = data['Person1_hand_raised','Person2_hand_raised','Person3_hand_raised','Person4_hand_raised']

    encoder = LabelEncoder()
    labels = encoder.fit_transform(yt).reshape(-1,1)

    # for folder in folders:
    #     # image_folder = os.path.join(folder, "images")
    #     csv = os.path.join(folder, "keypoints.csv")
    #     # label_csv = os.path.join(folder, "labels.csv")  # Assuming labels are in a separate file
        
    #     # Load images
    #     for img_name in sorted(os.listdir(folder)):
    #         img_path = os.path.join(folder, img_name)
    #         img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    #         img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # Normalize
    #         images.append(img_array)

    #     data = pd.concat([pd.read_csv(csv) for csv in csv], ignore_index=True)
    #     feature_columns = ["head", "neck", "left_ear", "right_ear", "left_shoulder", "left_elbow", "left_hand", "right_shoulder", "right_elbow", "right_hand"]
        

        
        # Load keypoints
        # keypoints = pd.read_csv(keypoints_csv)
        # keypoints_x.extend(keypoints.iloc[:, ::2].values)  # Extract x-coordinates
        # keypoints_y.extend(keypoints.iloc[:, 1::2].values)  # Extract y-coordinates

        # Load labels
        # labels.extend(pd.read_csv(label_csv).values.flatten())

    return np.array(images), np.array(X_x), np.array(X_y), np.array(labels)

# Hybrid model definition (unchanged from your original)
"""
def build_hybrid_model_old(input_image_shape, input_keypoints_dim):
    image_input = layers.Input(shape=input_image_shape)
    base_model = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet', input_tensor=image_input)
    base_model.trainable = False
    x = layers.GlobalAveragePooling2D()(base_model.output)
    
    # keypoints_input_x = layers.Input(shape=(input_keypoints_dim,), name="keypoints_x")
    # keypoints_input_y = layers.Input(shape=(input_keypoints_dim,), name="keypoints_y")
        
    x_branch = layers.Dense(128, activation='relu')
    # x_branch = layers.Dropout(0.3)(x_branch)
    # y_branch = layers.Dense(64, activation='relu')(keypoints_input_y)
    # y_branch = layers.Dropout(0.3)(y_branch)
    
    # combined_keypoints = layers.Concatenate()([X_x, X_y])
    combined_features = layers.concatenate([x, combined_keypoints])
    
    fc = layers.Dense(128, activation='relu')(combined_features)
    # fc = layers.Dropout(0.3)(fc)
    fc = layers.Dense(64, activation='relu')(fc)
    fc = layers.Dropout(0.3)(fc)
    output = layers.Dense(1, activation='sigmoid', name="Output")(fc)
    
    model = models.Model(inputs=[image_input, input_keypoints_dim], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


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
    keypoints_features = layers.Dropout(0.3)(keypoints_features)

    # Concatenate image and keypoint features
    combined_features = layers.concatenate([image_features, keypoints_features])

    # Fully connected layers for classification
    x = layers.Dense(128, activation='relu')(combined_features)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(4, activation='softmax', name="output")(x)

    # Build and compile model
    model = models.Model(inputs={"image_input": image_input, "keypoints_input": keypoints_input}, outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, images_folder, batch_size, target_image_size=(1280, 720)):
        self.data = data
        self.images_folder = images_folder
        self.batch_size = batch_size
        self.target_image_size = target_image_size
        self.indices = np.arange(len(self.data))
    
    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        images = []
        keypoints = []
        labels = []
        
        for i in batch_indices:
            row = self.data.iloc[i]
            # Load and preprocess image
            # image_path = self.images_folder + row['frame']
            image_path = f"{self.images_folder}/frame_{int(row['frame']):04d}.jpg"
            image = tf.keras.utils.load_img(image_path, target_size=self.target_image_size)
            image = tf.keras.utils.img_to_array(image) / 255.0  # Normalize to [0, 1]
            images.append(image)
            
            # Load keypoints
            keypoints.append(row[1:-1].values.astype(np.float32))  # Skip filename and label
            
            # Load label
            labels.append(row['label'])
        
        images = np.array(images)
        keypoints = np.array(keypoints)
        labels = np.array(labels)
        
        return {"image_input": images, "keypoints_input": keypoints}, labels
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)

'''
# Main script
# Specify folder paths for training and validation
train_folders = ["path_to_train_folder1", "path_to_train_folder2"]  # Add train folders
val_folders = ["path_to_val_folder1", "path_to_val_folder2"]  # Add validation folders

# Load training data
train_images, train_keypoints, train_labels = load_data_from_folders(train_folders)

# Load validation data
val_images, val_keypoints, val_labels = load_data_from_folders(val_folders)

# Define model input dimensions
input_image_shape = (224, 224, 3)  # Image dimensions
input_keypoints_dim = train_keypoints.shape[1]  # Number of keypoints

# Build the model
model = build_hybrid_model(input_image_shape, 10)

# Set up callbacks
checkpoint_callback = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    [train_images, train_keypoints],
    train_labels,
    validation_data=([val_images, val_keypoints], val_labels),
    batch_size=32,
    epochs=50,
    callbacks=[checkpoint_callback, early_stopping_callback]
)

# Load the best model
model.load_weights('best_model.h5')
'''
from sklearn.model_selection import train_test_split
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
input_keypoints_dim = 20
model = build_hybrid_model(input_image_shape, input_keypoints_dim)

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20
)

# Save the model
model.save("hybrid_model.h5")
'''
# Evaluate the model
loss, accuracy = model.evaluate([val_images, val_keypoints], val_labels)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")

# Make predictions
predictions = model.predict([val_images, val_keypoints])
print(predictions)
'''