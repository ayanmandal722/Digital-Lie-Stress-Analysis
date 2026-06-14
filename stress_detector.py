import os
import cv2
import numpy as np
import pickle
import librosa
import tensorflow as tf

class FacialStressDetector:
    def __init__(self, model_path='models/face_cnn.keras'):
        self.model_path = model_path
        self.cnn_model = None
        self.classes = ['anger', 'fear', 'happy', 'sad', 'neutral']
        
        # Load CNN model
        if os.path.exists(model_path):
            try:
                self.cnn_model = tf.keras.models.load_model(model_path)
                print(f"Loaded facial CNN model from {model_path}")
            except Exception as e:
                print(f"Error loading facial CNN model: {e}")
        else:
            print(f"Facial CNN model not found at {model_path}. Run model_trainer.py first.")
            
        # Load OpenCV cascades dynamically
        # Haar cascades are bundled inside cv2 installation and located via cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'))
        self.eye_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_eye.xml'))
        self.smile_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_smile.xml'))

    def analyze_frame(self, frame):
        """
        Analyze a single image frame (BGR format) and return stress score and details.
        """
        if self.face_cascade.empty():
            return {'stress_score': 0.35, 'emotion': 'neutral', 'details': {'cnn_probs': {}, 'smile_detected': False, 'eyes_detected': 0}}
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            return None  # No face detected in this frame
            
        # Analyze the largest face (by area)
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        (x, y, w, h) = faces[0]
        
        face_roi_gray = gray[y:y+h, x:x+w]
        
        # 1. Feature Heuristics (Smile and Eye detection)
        smile_detected = False
        if not self.smile_cascade.empty():
            # Smiles are located in the lower region of the face
            roi_lower_gray = face_roi_gray[int(h*0.55):h, :]
            smiles = self.smile_cascade.detectMultiScale(roi_lower_gray, scaleFactor=1.6, minNeighbors=20, minSize=(20, 20))
            if len(smiles) > 0:
                smile_detected = True
                
        eyes_detected = 0
        if not self.eye_cascade.empty():
            # Eyes are located in the upper region of the face
            roi_upper_gray = face_roi_gray[0:int(h*0.55), :]
            eyes = self.eye_cascade.detectMultiScale(roi_upper_gray, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15))
            eyes_detected = len(eyes)
            
        # 2. CNN Model Inference
        cnn_probs = {c: 0.0 for c in self.classes}
        cnn_probs['neutral'] = 1.0
        emotion_pred = 'neutral'
        cnn_stress_score = 0.2  # default neutral
        
        if self.cnn_model is not None:
            try:
                # Prepare face crop for the 48x48 CNN
                crop = cv2.resize(face_roi_gray, (48, 48))
                crop = crop.astype(np.float32) / 255.0
                crop = crop.reshape(1, 48, 48, 1)
                
                preds = self.cnn_model.predict(crop, verbose=0)[0]
                cnn_probs = {self.classes[i]: float(preds[i]) for i in range(len(self.classes))}
                
                # Get the top predicted emotion
                emotion_pred = self.classes[np.argmax(preds)]
                
                # Formula to map predicted emotion probabilities to a stress index [0, 1]
                # High stress: Anger (0.9), Fear (0.95)
                # Moderate stress: Sad (0.7)
                # Baseline: Neutral (0.2), Happy (0.0)
                cnn_stress_score = (
                    cnn_probs.get('anger', 0.0) * 0.9 +
                    cnn_probs.get('fear', 0.0) * 0.95 +
                    cnn_probs.get('sad', 0.0) * 0.7 +
                    cnn_probs.get('neutral', 0.0) * 0.2
                )
            except Exception as e:
                print(f"Error in CNN prediction: {e}")
                
        # 3. Decision-level heuristics blending
        final_stress_score = cnn_stress_score
        
        # Smile lowers stress score significantly and increases relaxation index
        if smile_detected:
            final_stress_score = final_stress_score * 0.2
            emotion_pred = 'happy'
        # Wide eyes detected in fear context increases stress estimation
        elif emotion_pred == 'fear' and eyes_detected >= 2:
            final_stress_score = min(1.0, final_stress_score * 1.15)
            
        final_stress_score = float(np.clip(final_stress_score, 0.0, 1.0))
        
        return {
            'stress_score': final_stress_score,
            'emotion': emotion_pred,
            'bbox': [int(x), int(y), int(w), int(h)],
            'details': {
                'cnn_probs': cnn_probs,
                'smile_detected': bool(smile_detected),
                'eyes_detected': int(eyes_detected)
            }
        }

    def analyze_video_file(self, video_path):
        """
        Analyze an uploaded video file frame-by-frame and compute average stress stats.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                'stress_score': 0.35,
                'emotion': 'neutral',
                'detected': False,
                'message': 'Could not open video file.'
            }
            
        scores = []
        emotions = []
        frame_count = 0
        sampled_count = 0
        max_frames = 120  # Limit frame parsing for fast dashboard feedback
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Sample every 3rd frame to reduce overhead
            if frame_count % 3 == 0:
                result = self.analyze_frame(frame)
                if result:
                    scores.append(result['stress_score'])
                    emotions.append(result['emotion'])
                    sampled_count += 1
                    
            frame_count += 1
            
        cap.release()
        
        if len(scores) == 0:
            return {
                'stress_score': 0.2,
                'emotion': 'neutral',
                'detected': False,
                'message': 'No face detected in video clip.'
            }
            
        avg_score = float(np.mean(scores))
        
        # Get the dominant facial emotion
        from collections import Counter
        dominant_emotion = Counter(emotions).most_common(1)[0][0]
        
        return {
            'stress_score': avg_score,
            'emotion': dominant_emotion,
            'detected': True,
            'frame_count': frame_count,
            'sampled_count': sampled_count
        }

class VocalStressDetector:
    def __init__(self, model_path='models/voice_classifier.pkl'):
        self.model_path = model_path
        self.voice_model = None
        self.scaler = None
        
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.voice_model = data['model']
                    self.scaler = data['scaler']
                print(f"Loaded vocal stress classifier from {model_path}")
            except Exception as e:
                print(f"Error loading vocal model: {e}")
        else:
            print(f"Vocal model not found at {model_path}. Run model_trainer.py first.")

    def analyze_audio_file(self, audio_path):
        """
        Extract MFCC features from an audio file using Librosa and run predictions.
        """
        if self.voice_model is None or self.scaler is None:
            return {'stress_score': 0.35, 'message': 'Vocal model not loaded.'}
            
        try:
            # Load audio using Librosa
            y, sr = librosa.load(audio_path, sr=16000)
            
            duration = librosa.get_duration(y=y, sr=sr)
            if duration < 0.2:
                return {'stress_score': 0.2, 'message': 'Audio clip is too short.'}
                
            # Extract 20 MFCC bands
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            
            # Create the 40-feature vector matching training (mean and standard deviation)
            features = np.hstack([mfcc_mean, mfcc_std])
            
            # Run inference
            scaled_features = self.scaler.transform(features.reshape(1, -1))
            prob = self.voice_model.predict_proba(scaled_features)[0]
            
            # Probability of Class 1 (Stressed)
            stress_score = float(prob[1])
            
            # Extract simple visualization vectors for the UI (spectrogram profile and average pitch profile)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            centroid_mean = float(np.mean(spectral_centroids))
            
            # Construct a spectrogram vector (average power across the 20 MFCC bands)
            spectrogram_data = [float(np.mean(np.abs(mfccs[i, :]))) for i in range(20)]
            
            return {
                'stress_score': stress_score,
                'duration': float(duration),
                'features': {
                    'mfcc_mean': [float(x) for x in mfcc_mean],
                    'centroid_mean': centroid_mean,
                    'spectrogram_data': spectrogram_data
                }
            }
        except Exception as e:
            print(f"Error in vocal analysis: {e}")
            return {'stress_score': 0.35, 'message': f"Analysis error: {str(e)}"}

class MultimodalFusionDetector:
    def __init__(self, face_model_path='models/face_cnn.keras', voice_model_path='models/voice_classifier.pkl'):
        self.face_detector = FacialStressDetector(face_model_path)
        self.voice_detector = VocalStressDetector(voice_model_path)
        
    def fuse_predictions(self, face_result, voice_result):
        """
        Performs Decision Late Fusion.
        Accuracy performance metrics from paper:
        - Facial only: 72%
        - Voice only: 78%
        - Multimodal: 86%
        Weight selection reflects voice reliability: w_voice = 0.55, w_face = 0.45.
        """
        has_face = face_result is not None and 'stress_score' in face_result and face_result.get('detected', True)
        has_voice = voice_result is not None and 'stress_score' in voice_result and 'message' not in voice_result
        
        w_face = 0.45
        w_voice = 0.55
        
        if has_face and has_voice:
            face_score = face_result['stress_score']
            voice_score = voice_result['stress_score']
            fused_score = w_face * face_score + w_voice * voice_score
            modality_used = "multimodal"
        elif has_face:
            fused_score = face_result['stress_score']
            modality_used = "facial_only"
        elif has_voice:
            fused_score = voice_result['stress_score']
            modality_used = "voice_only"
        else:
            fused_score = 0.2  # default baseline
            modality_used = "default"
            
        fused_score = float(np.clip(fused_score, 0.0, 1.0))
        
        # Map stress index to qualitative category
        if fused_score < 0.35:
            stress_level = "Relaxed"
            stress_description = "Low stress levels. You appear calm and relaxed."
        elif fused_score < 0.65:
            stress_level = "Mild Stress"
            stress_description = "Moderate stress levels detected. Consider taking a short break or drinking water."
        else:
            stress_level = "Highly Stressed"
            stress_description = "High stress levels detected! Try the deep breathing exercise or grounding techniques."
            
        return {
            'fused_stress_score': fused_score,
            'stress_level': stress_level,
            'description': stress_description,
            'modality_used': modality_used,
            'face_score': face_result['stress_score'] if has_face else None,
            'face_emotion': face_result.get('emotion') if has_face else None,
            'voice_score': voice_result['stress_score'] if has_voice else None
        }
  