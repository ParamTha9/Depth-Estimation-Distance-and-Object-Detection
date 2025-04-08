import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO

# Define MiDaS transformation (for MiDaS Small)
midas_transforms = transforms.Compose([
    transforms.Resize((384, 384)),  # Resize to the expected input size for MiDaS Small
    transforms.ToTensor(),  # Convert PIL image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Load MiDaS model
midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
midas_model.eval()

# Known object dimensions in meters (e.g., a bottle with height = 0.30m)
KNOWN_OBJECT_HEIGHT = 0.30  # Real-world height of the reference object in meters

# Camera parameters
FOCAL_LENGTH = 600  # Approximate focal length of the webcam in pixels
# Note: Focal length can be calibrated more precisely using a checkerboard calibration process.

# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")  # Load YOLOv8 model

# Target class index for "bottle" (COCO dataset class index)
TARGET_CLASS = 39

# Initialize distance buffer for smoothing
distance_buffer = []
buffer_size = 10  # Number of frames for temporal smoothing

# Webcam setup
cap = cv2.VideoCapture(0)  # Change index if you have multiple cameras

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster processing
    frame = cv2.resize(frame, (640, 480))

    # Convert frame to RGB for YOLOv8 processing
    input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # YOLOv8 object detection
    results = yolo_model(input_frame, stream=True)
    detections = []
    for result in results:
        boxes = result.boxes.xyxy  # Bounding box coordinates
        confidences = result.boxes.conf
        classes = result.boxes.cls

        for i, box in enumerate(boxes):
            # Filter for "bottle" class
            if int(classes[i]) == TARGET_CLASS:
                x1, y1, x2, y2 = map(int, box)
                detections.append((x1, y1, x2, y2, int(classes[i]), confidences[i]))

    # MiDaS depth estimation
    pil_frame = Image.fromarray(input_frame)  # Convert numpy array to PIL image
    transformed_frame = midas_transforms(pil_frame).unsqueeze(0)  # Apply MiDaS transformation

    with torch.no_grad():
        depth_map = midas_model(transformed_frame)  # Forward pass to get depth map

        # Resize depth map to match the input frame size (height, width)
        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),  # Add channel dimension if missing
            size=(frame.shape[0], frame.shape[1]),
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()

    # Normalize depth for visualization
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Process YOLO detections for "bottle" class
    for (x1, y1, x2, y2, cls, conf) in detections:
        # Crop the depth map to the bounding box
        depth_crop = depth_map[y1:y2, x1:x2]

        # Filter invalid depth values
        valid_depths = depth_crop[(depth_crop > 0) & (depth_crop < np.percentile(depth_crop, 99))]
        if len(valid_depths) == 0:
            continue  # Skip if no valid depths

        # Use the median depth for stability
        obj_depth = np.median(valid_depths)

        # Calculate apparent object height in pixels
        apparent_height = y2 - y1  # Bounding box height in pixels

        if apparent_height > 0:  # Avoid division by zero
            # Distance using focal length formula
            distance_focal = (KNOWN_OBJECT_HEIGHT * FOCAL_LENGTH) / apparent_height

            # Distance using depth map scaling factor
            scaling_factor = KNOWN_OBJECT_HEIGHT / apparent_height
            distance_depth = obj_depth * scaling_factor

            # Use focal length-based distance for better accuracy
            real_distance = distance_focal

            # Add the distance to the buffer for smoothing
            distance_buffer.append(real_distance)
            if len(distance_buffer) > buffer_size:
                distance_buffer.pop(0)

            # Calculate the smoothed distance
            stable_distance = np.mean(distance_buffer)

            # Label the object with the stabilized distance
            label = f"Bottle: Distance: {stable_distance:.2f}m"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display depth map and frame
    cv2.imshow("Depth Map", depth_normalized)
    cv2.imshow("YOLO + Depth Estimation", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
