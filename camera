import cv2
import tensorflow as tf
import numpy as np

# Load the pre-trained object detection model
model = tf.saved_model.load('path/to/model')

# Initialize the camera
camera = cv2.VideoCapture(0)

# Define the class labels
labels = ['class1', 'class2', 'class3', ...]

# Loop through frames from the camera
while True:
    # Capture a frame from the camera
    ret, frame = camera.read()

    # Convert the image to a format suitable for input to the model
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = tf.convert_to_tensor(image)
    image = tf.expand_dims(image, 0)

    # Perform object detection on the image using the pre-trained model
    output = model(image)
    boxes = output['detection_boxes'][0].numpy()
    scores = output['detection_scores'][0].numpy()
    classes = output['detection_classes'][0].numpy().astype(np.int32)

    # Filter out detections with low confidence scores
    detections = []
    for i in range(len(scores)):
        if scores[i] > 0.5:
            detections.append((boxes[i], labels[classes[i]]))

    # Draw the detections on the image and display it
    for box, label in detections:
        ymin, xmin, ymax, xmax = box
        xmin = int(xmin * frame.shape[1])
        xmax = int(xmax * frame.shape[1])
        ymin = int(ymin * frame.shape[0])
        ymax = int(ymax * frame.shape[0])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(frame, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Object Detection', frame)

    # Exit on key press
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
camera.release()
cv2.destroyAllWindows()
