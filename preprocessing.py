from collections import deque
import cv2
import numpy as np
import pyautogui
import pygetwindow as gw
import matplotlib.pyplot as plt

# Game details
game_title = 'Getting Over It'
window = gw.getWindowsWithTitle(game_title)[0]
x, y, width, height = window.left, window.top, window.width, window.height

# Frame stacking
stack_size = 4  # Adjust this based on your game's temporal dynamics
frame_stack = deque(maxlen=stack_size)

# Define the color for each object in the abstracted version
PLAYER_COLOR = np.array([255, 0, 0])  # Red
obstacle_color = np.array([0, 255, 0])  # Green
environment_color = np.array([0, 0, 255])  # Blue

# Data augmentation
noise_std = 0.1  # Adjust this based on the scale and nature of your data


def add_noise(frame):
    noise = np.random.normal(0, noise_std, frame.shape)
    frame += noise
    frame = np.clip(frame, 0, 1)  # Ensure the added noise doesn't cause data to go out of range
    return frame


def rotate_frame(frame, angle):
    # Get image height and width
    (h, w) = frame.shape[:2]

    # Define the center of the image
    center = (w / 2, h / 2)

    # Perform the rotation
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(frame, m, (w, h))

    return rotated


def visualize_mask(mask, title):
    plt.figure(figsize=(10, 10))
    plt.imshow(mask, cmap='hot')
    plt.title(title)
    plt.show()


def visualize_histogram(data, title):
    plt.figure(figsize=(10, 10))
    plt.hist(data.ravel(), bins=256, color='orange', )
    plt.title(title)
    plt.show()


def visualize_frame_stack(frame_stack):
    for i, frame in enumerate(frame_stack):
        # Transpose the frame back to its original shape
        frame = np.transpose(frame, (1, 2, 0))

        # Rescale the frame to the range 0-255
        frame = (frame * 255).astype(np.uint8)

        plt.figure(figsize=(10, 10))
        plt.imshow(frame, cmap='gray')
        plt.title(f'Frame {i+1}')
        plt.show()


# Define the size for the low-res, pixelated version
low_res_size = (64, 64)  # Adjust this based on your needs


def preprocess_frame(frame, frame_stack):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the color range for each object
    player_range = (np.array([0, 100, 100]), np.array([10, 255, 255]))  # Red range
    obstacle_range = (np.array([35, 100, 100]), np.array([85, 255, 255]))  # Green range
    environment_range = (np.array([110, 100, 100]), np.array([130, 255, 255]))  # Blue range

    # Create a binary mask for each object
    player_mask = cv2.inRange(hsv, player_range[0], player_range[1])
    obstacle_mask = cv2.inRange(hsv, obstacle_range[0], obstacle_range[1])
    environment_mask = cv2.inRange(hsv, environment_range[0], environment_range[1])

    # Combine the masks into a single image
    preprocessed = np.zeros_like(frame)
    player_color = (255, 255, 255)  # example RGB color for the player
    preprocessed[player_mask > 0] = player_color
    preprocessed[obstacle_mask > 0] = obstacle_color
    preprocessed[environment_mask > 0] = environment_color

    # Resize the preprocessed frame to a lower resolution
    preprocessed = cv2.resize(preprocessed, low_res_size)

    # Normalize the preprocessed frame
    preprocessed = preprocessed / 255.0

    # Add noise for data augmentation
    preprocessed = add_noise(preprocessed)

    # Transpose the frame to have channels as the first dimension
    preprocessed = np.transpose(preprocessed, (2, 0, 1))

    # Add the preprocessed frame to the frame stack
    frame_stack.append(preprocessed)

    # Stack the frames along the channel dimension
    if len(frame_stack) < stack_size:
        # If stack is not full, pad with zeros
        stacked_frames = np.concatenate(
            [np.zeros(((stack_size - len(frame_stack)) * preprocessed.shape[0],
                       preprocessed.shape[1], preprocessed.shape[2])), *frame_stack], axis=0)
    else:
        # If stack is full or overfull, drop the oldest frames and use the newest ones
        stacked_frames = np.concatenate(list(frame_stack), axis=0)[-stack_size * preprocessed.shape[0]:]

    return stacked_frames, frame_stack


def display_preprocessed_frame(frame):
    # Convert frame to a NumPy array if it's not already
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)

    # Undo the normalization and transpose back to the original shape
    display_frame = (frame * 255.0).astype(np.uint8)
    display_frame = np.transpose(display_frame, (1, 2, 0))

    # Use only the first three channels of the image
    if display_frame.shape[2] > 3:
        display_frame = display_frame[:, :, :3]

    # Create a figure with subplots
    fig, ax = plt.subplots(figsize=(10, 10))

    # Show the frame with a color map
    im = ax.imshow(display_frame, cmap='hot')

    # Add a color bar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Pixel Intensity')

    # Add a title
    ax.set_title('Preprocessed Frame')

    # Display the figure
    plt.show()


# Capture a screenshot of the game
while True:
    game_screen = pyautogui.screenshot(region=(x, y, width, height))
    game_screen_np = np.array(game_screen)
    preprocessed_frame, frame_stack = preprocess_frame(game_screen_np, frame_stack)

    display_preprocessed_frame(preprocessed_frame)  # Show the processed frame

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
