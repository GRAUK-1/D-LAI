import time
import cv2
import numpy as np
import pyautogui
import pygetwindow as gw
import torch
from ai import GettingOverItAI
from constants import GAME_TITLE, BATCH_SIZE, SAVE_INTERVAL, BUFFER_MAXLEN
from models import SimpleModel, CuriosityModel
from preprocessing import preprocess_frame
from buffer import ReplayBuffer
from torch import optim

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
    curiosity_model = CuriosityModel()
    ai = GettingOverItAI(model, curiosity_model)
    replay_buffer = ReplayBuffer(BUFFER_MAXLEN)

    last_save_time = time.time()

    # Define an optimizer
    optimizer = optim.Adam(model.parameters())

    while True:
        try:
            # Capture a screenshot of the game
            game_screen = pyautogui.screenshot(region=(x, y, width, height))
            game_screen_np = np.array(game_screen)

            # Preprocess the game screen
            preprocessed_screen_np = preprocess_frame(game_screen_np)

            # Convert the 12-channels image to multiple 3-channels (RGB) images for visualization
            for i in range(0, preprocessed_screen_np.shape[2], 3):  # Assume the shape is (height, width, 12)
                visualization = preprocessed_screen_np[:, :, i:i + 3]  # Take three channels at a time

                # Scale the visualization to range 0-255 if it's not already in that range
                visualization = cv2.normalize(visualization, None, alpha=0.0, beta=255.0, norm_type=cv2.NORM_MINMAX,
                                              dtype=0)
                cv2.imshow(f'AI View Channels {i + 1}-{i + 3}', visualization)
                cv2.waitKey(1)  # Refresh the display every 1 millisecond

            # Convert the numpy array to a tensor and move it to the device
            state_tensor = torch.from_numpy(preprocessed_screen_np).float().to(device)

            # Convert the tensor back to a numpy array
            state = state_tensor.cpu().numpy()

            # Feed the preprocessed screen into the AI model
            predicted_action = ai.predict(state)

            # Save the current screen and action to the buffer
            replay_buffer.push(preprocessed_screen_np, predicted_action, 0, 0, False)

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
            break

        finally:
            cv2.destroyAllWindows()

    ai.save_model()


if __name__ == "__main__":
    main()
