import os
import glob
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def extract_mfcc_features(audio_path, n_mfcc=13, n_fft=2048, hop_length=512):
    try:
        audio_data, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return None

    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return np.mean(mfccs.T, axis=0)

def create_dataset(directory, label):
    X, y = [], []
    audio_files = glob.glob(os.path.join(directory, "*.wav"))
    for audio_path in audio_files:
        mfcc_features = extract_mfcc_features(audio_path)
        if mfcc_features is not None:
            X.append(mfcc_features)
            y.append(label)
        else:
            print(f"Skipping audio file {audio_path}")

    print("Number of samples in", directory, ":", len(X))
    print("Filenames in", directory, ":", [os.path.basename(path) for path in audio_files])
    return X, y

def build_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

def train_model(X, y):
    X = np.array(X)
    y = np.array(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = build_model(X_train_scaled.shape[1])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_data=(X_test_scaled, y_test))
    
    model.save("audio_model.h5")
    
    y_pred = model.predict(X_test_scaled)
    y_pred = (y_pred > 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    confusion_mtx = confusion_matrix(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:")
    print(confusion_mtx)

def analyze_audio(input_audio_path):
    model = tf.keras.models.load_model("audio_model.h5")
    
    if not os.path.exists(input_audio_path):
        print("Error: The specified file does not exist.")
        return
    elif not input_audio_path.lower().endswith(".wav"):
        print("Error: The specified file is not a .wav file.")
        return

    mfcc_features = extract_mfcc_features(input_audio_path)

    if mfcc_features is not None:
        scaler = StandardScaler()
        mfcc_features_scaled = scaler.fit_transform(mfcc_features.reshape(1, -1))
        prediction = model.predict(mfcc_features_scaled)
        if prediction[0][0] < 0.5:
            print("The input audio is real.")
        else:
            print("The input audio is clone.")
    else:
        print("Error: Unable to process the input audio.")

def main():
    real_dir = r"dataset\real"
    clone_dir = r"dataset\clone"

    X_genuine, y_genuine = create_dataset(real_dir, label=0)
    X_deepclone, y_deepclone = create_dataset(clone_dir, label=1)

    X = X_genuine + X_deepclone
    y = y_genuine + y_deepclone

    train_model(X, y)

if __name__ == "__main__":
    main()

    user_input_file = input("Enter the path of the .wav file to analyze: ")
    analyze_audio(user_input_file)
