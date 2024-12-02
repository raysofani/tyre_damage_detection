"""
import cv2
import numpy as np


def process_image_file(model, image_path):

   # Processes the input image for prediction

    # Load the image using OpenCV
    img = cv2.imread(image_path)

    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize the image to the required input size (128x128)
    img = cv2.resize(img, (128, 128))

    # Normalize the image
    img = img.astype("float32") / 255.0

    # Add an extra dimension to match the model input shape (batch size of 1)
    img = np.expand_dims(img, axis=0)

    # Make prediction
    prediction = model.predict(img)

    # Assuming the model outputs the damage percentage
    damage_percentage = prediction[0][0]  # Adjust based on your model's output format

    # Enhanced damage assessment
    if damage_percentage < 0.1:
        damage_status = "Minimal Wear"
    elif damage_percentage < 0.3:
        damage_status = "Light Wear"
    elif damage_percentage < 0.5:
        damage_status = "Moderate Wear"
    elif damage_percentage < 0.7:
        damage_status = "Significant Wear"
    else:
        damage_status = "Severe Damage"

    print(f"Damage Assessment: {damage_status}")
    print(f"Damage Percentage: {damage_percentage * 100:.2f}%")


def analyze_with_image_or_webcam(model, input_type="webcam", image_path=None):

   # Analyzes an image or webcam stream based on the input type

    if input_type == "image":
        if image_path is not None:
            process_image_file(model, image_path)
        else:
            print("No image path provided.")
    elif input_type == "webcam":
        # Webcam detection logic (same as before)
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize and preprocess the frame for the model
            img = cv2.resize(frame, (128, 128))
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=0)

            # Make prediction
            prediction = model.predict(img)
            damage_percentage = prediction[0][0]

            # Display the resulting frame with damage percentage text
            cv2.putText(frame, f"Damage: {damage_percentage * 100:.2f}%", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Tire Damage Detection', frame)

            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the webcam and close any OpenCV windows
        cap.release()
        cv2.destroyAllWindows()
"""
import cv2
import numpy as np
import os


def process_image_file(model, image_path):

   # Processes the input image for prediction

    # Load the image using OpenCV
    img = cv2.imread(image_path)

    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize the image to the required input size (128x128)
    img = cv2.resize(img, (128, 128))

    # Normalize the image
    img = img.astype("float32") / 255.0

    # Add an extra dimension to match the model input shape (batch size of 1)
    img = np.expand_dims(img, axis=0)

    # Make prediction
    prediction = model.predict(img)

    # Assuming the model outputs the damage percentage
    damage_percentage = prediction[0][0]  # Adjust based on your model's output format

    # Enhanced damage assessment
    if damage_percentage < 0.1:
        damage_status = "Minimal Wear"
    elif damage_percentage < 0.3:
        damage_status = "Light Wear"
    elif damage_percentage < 0.5:
        damage_status = "Moderate Wear"
    elif damage_percentage < 0.7:
        damage_status = "Significant Wear"
    else:
        damage_status = "Severe Damage"

    print(f"Damage Assessment: {damage_status}")
    print(f"Damage Percentage: {damage_percentage * 100:.2f}%")




def process_video_file(model, video_path, output_path=None):
    """
    Processes a video file for continuous tire damage detection with enhanced debugging
    """
    # Validate video path
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    # Open video capture
    cap = cv2.VideoCapture(video_path)

    # Check video opening
    if not cap.isOpened():
        print(f"Critical Error: Cannot open video file {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video Details:")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total Frames: {total_frames}")

    # Output video writer
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    damage_history = []
    window_size = 30
    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Preprocess frame
        input_frame = cv2.resize(frame, (128, 128))
        input_frame = input_frame.astype("float32") / 255.0
        input_frame = np.expand_dims(input_frame, axis=0)

        # Predict damage
        prediction = model.predict(input_frame)
        damage_percentage = prediction[0][0]

        damage_history.append(damage_percentage)
        if len(damage_history) > window_size:
            damage_history.pop(0)

        avg_damage = np.mean(damage_history)

        # Damage status determination
        if avg_damage < 0.1:
            damage_status = "Minimal Wear"
            color = (0, 255, 0)  # Green
        elif avg_damage < 0.3:
            damage_status = "Light Wear"
            color = (0, 255, 255)  # Yellow
        elif avg_damage < 0.5:
            damage_status = "Moderate Wear"
            color = (0, 165, 255)  # Orange
        elif avg_damage < 0.7:
            damage_status = "Significant Wear"
            color = (0, 0, 255)  # Red
        else:
            damage_status = "Severe Damage"
            color = (0, 0, 128)  # Dark Red

        # Annotate frame
        cv2.putText(frame,
                    f"Damage: {avg_damage * 100:.2f}% - {damage_status}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2)

        # Show frame
        cv2.imshow('Tire Damage Video Analysis', frame)

        # Write output if specified
        if out is not None:
            out.write(frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        frame_count += 1

    # Cleanup
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

    print(f"\nProcessing complete. Frames processed: {frame_count}")
    print(f"Average Damage: {avg_damage * 100:.2f}%")
    print(f"Overall Status: {damage_status}")


def analyze_with_image_or_webcam(model, input_type="webcam", image_path=None, video_path=None, output_path=None):
    """
    Analyzes an image, webcam stream, or video based on the input type

    Args:
        model: Trained Keras model
        input_type (str): Type of input (image, webcam, video)
        image_path (str, optional): Path to input image
        video_path (str, optional): Path to input video
        output_path (str, optional): Path to save output video
    """
    if input_type == "image":
        # Existing image processing logic
        if image_path is not None:
            process_image_file(model, image_path)
        else:
            print("No image path provided.")

    elif input_type == "video":
        # New video processing logic
        if video_path is not None:
            process_video_file(model, video_path, output_path)
        else:
            print("No video path provided.")

    elif input_type == "webcam":
        # Existing webcam detection logic
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize and preprocess the frame for the model
            img = cv2.resize(frame, (128, 128))
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=0)

            # Make prediction
            prediction = model.predict(img)
            damage_percentage = prediction[0][0]

            # Display the resulting frame with damage percentage text
            cv2.putText(frame, f"Damage: {damage_percentage * 100:.2f}%", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Tire Damage Detection', frame)

            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the webcam and close any OpenCV windows
        cap.release()
        cv2.destroyAllWindows()