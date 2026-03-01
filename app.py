# save this as app.py
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import json

class HandGestureRecognizer:
    def __init__(self, model_path='hand_gesture_model.h5', 
                 tflite_path='hand_gesture_model.tflite',
                 labels_path='labels.json',
                 confidence_threshold=0.70):
        
        # Load labels
        with open(labels_path, 'r') as f:
            self.labels = json.load(f)
        
        # Set confidence threshold for unknown gesture detection
        self.confidence_threshold = confidence_threshold
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load model (use TFLite for faster inference)
        try:
            # Try TFLite first (faster)
            self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.use_tflite = True
            print("✓ Using TFLite model (fast inference)")
        except:
            # Fall back to Keras model
            self.model = tf.keras.models.load_model(model_path)
            self.use_tflite = False
            print("✓ Using Keras H5 model")
        
        print(f"✓ Confidence threshold set to {self.confidence_threshold}")
        print("  Gestures below this threshold will be classified as 'Unknown'")
    
    def extract_landmarks(self, image):
        """Extract hand landmarks from image"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw landmarks on image
            self.mp_drawing.draw_landmarks(
                image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # Extract coordinates
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            return np.array(landmarks), image
        
        return None, image
    
    def predict(self, landmarks):
        """Predict gesture from landmarks with unknown gesture detection"""
        landmarks = np.array(landmarks).reshape(1, -1).astype(np.float32)
        
        if self.use_tflite:
            # TFLite inference
            self.interpreter.set_tensor(self.input_details[0]['index'], landmarks)
            self.interpreter.invoke()
            prediction = self.interpreter.get_tensor(self.output_details[0]['index'])
        else:
            # Keras inference
            prediction = self.model.predict(landmarks, verbose=0)
        
        gesture_id = np.argmax(prediction)
        confidence = prediction[0][gesture_id]
        
        # If confidence is below threshold, classify as unknown
        if confidence < self.confidence_threshold:
            return "Unknown", confidence
        
        return self.labels[str(gesture_id)], confidence
    
    def recognize_from_frame(self, frame):
        """Process a single frame"""
        landmarks, annotated_frame = self.extract_landmarks(frame)
        
        if landmarks is not None:
            gesture, confidence = self.predict(landmarks)
            return gesture, confidence, annotated_frame
        
        return None, 0, annotated_frame

def main():
    # Initialize recognizer with adjustable confidence threshold
    # Lower threshold (e.g., 0.6) = more lenient, fewer unknowns
    # Higher threshold (e.g., 0.8) = more strict, more unknowns
    recognizer = HandGestureRecognizer(confidence_threshold=0.70)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\nGesture Recognition Started!")
    print("Press 'q' to quit")
    print("Press 'r' to reset gesture history")
    print("Show your hand to the camera...")
    print("Trained gestures: Thumbs Up, Open Palm, Peace, Point, OK, Fist")   
    print("Other gestures will be shown as 'Unknown'\n")
    
    # Variables for smoothing predictions
    gesture_history = []
    history_length = 5
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Recognize gesture
        gesture, confidence, processed_frame = recognizer.recognize_from_frame(frame)
        
        # Smooth predictions (reduce flickering)
        if gesture:
            gesture_history.append(gesture)
            if len(gesture_history) > history_length:
                gesture_history.pop(0)
            
            # Get most common gesture in history
            from collections import Counter
            if gesture_history:
                smoothed_gesture = Counter(gesture_history).most_common(1)[0][0]
            else:
                smoothed_gesture = gesture
            
            # Display result with different colors
            text = f"{smoothed_gesture} ({confidence:.2f})"
            
            # Color coding:
            # Green = high confidence known gesture
            # Orange = medium confidence known gesture  
            # Red = unknown gesture
            if smoothed_gesture == "Unknown":
                color = (0, 0, 255)  # Red for unknown
            elif confidence > 0.8:
                color = (0, 255, 0)  # Green for high confidence
            else:
                color = (0, 165, 255)  # Orange for medium confidence
            
            cv2.putText(processed_frame, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Add confidence bar
            bar_width = int(confidence * 200)
            cv2.rectangle(processed_frame, (10, 45), (10 + bar_width, 55), color, -1)
        
        # Display instructions
        cv2.putText(processed_frame, "Trained: Thumbs Up | Open Palm | Peace | Point | OK | Fist", (10, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(processed_frame, "Press 'q' to quit | 'r' to reset", (10, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show the frame
        cv2.imshow('Hand Gesture Recognition', processed_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            gesture_history = []
            print("Gesture history reset")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nGesture recognition stopped.")

if __name__ == "__main__":
    main()