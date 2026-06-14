import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

# 1. Train Audio/Voice Stress Classifier
def train_voice_model():
    print("Generating synthetic audio features for voice stress model...")
    # Generate 500 samples, 40 features each (mean & std of 20 MFCCs)
    np.random.seed(42)
    n_samples = 500
    n_features = 40
    
    # Class 0: Relaxed, Class 1: Stressed
    # Stressed speech typically features higher energy, jitter, and spectral shifts.
    # We simulate this with distinct Gaussian distributions for MFCC statistics.
    X_relaxed = np.random.normal(loc=0.0, scale=1.0, size=(n_samples // 2, n_features))
    X_stressed = np.random.normal(loc=1.2, scale=1.4, size=(n_samples // 2, n_features))
    
    X = np.vstack([X_relaxed, X_stressed])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    
    # Shuffle
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    accuracy = clf.score(X_test_scaled, y_test)
    print(f"Voice Stress Classifier trained. Test Accuracy: {accuracy * 100:.2f}%")
    
    # Save model and scaler
    with open('models/voice_classifier.pkl', 'wb') as f:
        pickle.dump({'model': clf, 'scaler': scaler}, f)
    print("Saved voice model to models/voice_classifier.pkl")

# 2. Train Facial Expression CNN
def train_face_model():
    print("Generating synthetic face images for Keras CNN emotion classifier...")
    # Classes: 0: Anger, 1: Fear, 2: Happy, 3: Sad, 4: Neutral
    classes = ['anger', 'fear', 'happy', 'sad', 'neutral']
    num_classes = len(classes)
    img_size = 48
    samples_per_class = 200
    
    X = []
    y = []
    
    # We will generate simulated facial drawing matrices (eyes + expression mouth shape)
    # for each emotion class, adding Gaussian noise to simulate face variations.
    for class_idx, label in enumerate(classes):
        for _ in range(samples_per_class):
            img = np.zeros((img_size, img_size), dtype=np.uint8) + 128  # mid-gray background
            
            # Draw eyes (always present)
            # Left eye
            img[15:18, 12:15] = 30
            # Right eye
            img[15:18, 32:35] = 30
            
            # Draw expressions
            if label == 'happy':
                # Smile (upward curve/bottom line)
                img[32:34, 18:30] = 30
                img[30:32, 16:18] = 30
                img[30:32, 30:32] = 30
            elif label == 'sad':
                # Frown (downward curve)
                img[30:32, 18:30] = 30
                img[32:34, 16:18] = 30
                img[32:34, 30:32] = 30
            elif label == 'anger':
                # Angry eyebrows (slanting down towards center)
                img[12, 10:16] = 30
                img[13, 13:16] = 30
                img[12, 32:38] = 30
                img[13, 32:35] = 30
                # Tight straight mouth
                img[31:33, 18:30] = 30
            elif label == 'fear':
                # Raised eyebrows
                img[10:12, 11:15] = 30
                img[10:12, 33:37] = 30
                # Small open mouth (gasp)
                img[30:34, 21:27] = 30
            elif label == 'neutral':
                # Straight line mouth
                img[31:32, 18:30] = 30
                
            # Add Gaussian noise to create variation
            noise = np.random.normal(0, 15, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            X.append(img)
            y.append(class_idx)
            
    X = np.array(X, dtype=np.float32) / 255.0
    X = X.reshape(-1, img_size, img_size, 1)
    y = np.array(y)
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define ConvNet
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
                  
    print("Training Face CNN model...")
    model.fit(X_train, y_train, epochs=8, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Face CNN trained. Test Accuracy: {acc * 100:.2f}%")
    
    # Save the modern Keras model format
    model.save('models/face_cnn.keras')
    print("Saved face CNN model to models/face_cnn.keras")

if __name__ == '__main__':
    train_voice_model()
    train_face_model()
  