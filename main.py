import time
import cv2
import pygetwindow as gw
import torch
from ai import GettingOverItAI
from constants import GAME_TITLE, BATCH_SIZE, SAVE_INTERVAL, BUFFER_MAXLEN
from models import SimpleModel, CuriosityModel
from preprocessing import preprocess_frame
from buffer import ReplayBuffer
from torch import optim
from collections import deque
import numpy as np

# Initialize the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Game details
window = gw.getWindowsWithTitle(GAME_TITLE)[0]
x, y, width, height = window.left, window.top, window.width, window.height

# The number of color channels in the input images
# This is 3 for RGB images or 1 for grayscale images
channels = 3  # Replace with the number of color channels in your preprocessed images

# The number of frames stacked together
STACK_SIZE = 4  # This is a common value, but you should replace it with the actual number you're using

input_shape = (STACK_SIZE * channels, height, width)
model = SimpleModel(input_shape)


def main():
    # Initialize the curiosity model and AI
    curiosity_model = CuriosityModel()
    ai = GettingOverItAI(model, curiosity_model)

    # Initialize the replay buffer
    replay_buffer = ReplayBuffer(BUFFER_MAXLEN)

    # Define an optimizer
    optimizer = optim.Adam(model.parameters())

    # Initialize the video capture
    cap = cv2.VideoCapture(GAME_TITLE)

    # Initialize frame stack
    frame_stack = deque(maxlen=STACK_SIZE)

    last_save_time = time.time()

    try:
        while cap.isOpened():
            # Capture a frame from the video
            ret, frame = cap.read()

            if not ret:
                break

            # Preprocess the frame
            preprocessed_frame = preprocess_frame(frame)

            # Add the preprocessed frame to the frame stack
            frame_stack.append(preprocessed_frame)

            # If the frame stack is not yet full, skip this iteration
            if len(frame_stack) < STACK_SIZE:
                continue

            # Convert the frame stack to a numpy array
            frame_stack_np = np.array(frame_stack)

            # Convert the numpy array to a tensor and move it to the device
            state_tensor = torch.from_numpy(frame_stack_np).float().to(device)

            # Convert the tensor back to a numpy array
            state = state_tensor.cpu().numpy()

            # Feed the preprocessed frame into the AI model
            predicted_action = ai.predict(state)

            # Save the current frame and action to the buffer
            replay_buffer.push(frame_stack_np, predicted_action, 0, 0, False)

            if len(replay_buffer) >= BATCH_SIZE:
                # When replay buffer has enough experiences
                state, action, reward, next_state, done = replay_buffer.sample(BATCH_SIZE)

                # Set the gradients to zero
                optimizer.zero_grad()

                # Compute the loss
                loss = ai.compute_loss(state, action, reward, next_state, done)

                # Backpropagation
                loss.backward()

                # Optimization step
                optimizer.step()

                # Save the model at a regular interval
                if time.time() - last_save_time >= SAVE_INTERVAL:
                    ai.save_model()
                    last_save_time = time.time()

            time.sleep(0.1)

    except Exception as e:

        print(f"An error occurred: {e}")

    finally:

        # Release the video capture and close all windows

        cap.release()

        cv2.destroyAllWindows()

        # Save the model

        ai.save_model()


if __name__ == "__main__":
    main()
