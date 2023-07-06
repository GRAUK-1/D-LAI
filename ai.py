import logging
import cv2
import numpy as np
import pyautogui
import torch
import torch.nn as nn
import torch.optim as optim
from skopt import gp_minimize
from torch import device
from torchvision.transforms import ToTensor
from buffer import ReplayBuffer
from constants import BUFFER_MAXLEN, EPSILON, EPSILON_DECAY, EPSILON_MIN, OUTPUT_SIZE, BATCH_SIZE
from preprocessing import preprocess_frame, PLAYER_COLOR

logging.basicConfig(filename='ai.log', level=logging.ERROR)


# Initialize the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GettingOverItAI:
    def __init__(self, model, curiosity_model):
        self.model = model
        self.curiosity_model = curiosity_model
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00025)
        self.curiosity_optimizer = optim.Adam(self.curiosity_model.parameters(), lr=0.00025)
        self.transform = ToTensor()
        self.total_reward = 0.0
        self.previous_position = None
        self.buffer = ReplayBuffer(BUFFER_MAXLEN)
        self.epsilon = EPSILON
        self.steps_since_progress = 0
        self.max_position = None
        self.previous_mouse_position = None
        self.previous_position = 0.0
        self.gamma = 0.99  # Discount factor for future rewards

    @staticmethod
    def get_screen_shot():
        try:
            # Capture the entire screen
            screen = np.array(pyautogui.screenshot())
            # Convert the image from BGR to RGB color space
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            return screen
        except Exception as e:
            logging.error(f'Error capturing screen: {e}')

    @staticmethod
    def perform_mouse_movement(action):
        try:
            # Get the current mouse position
            current_position = np.array(pyautogui.position())
            # Calculate the new mouse position
            new_position = current_position + action
            # Ensure the new position is within the screen bounds
            new_position = np.clip(new_position, 0, np.array(pyautogui.size()))
            # Move the mouse to the new position
            pyautogui.moveTo(new_position[0], new_position[1], duration=0.1)
        except Exception as e:
            logging.error(f'Error performing mouse movement: {e}')

    def simulate_game(self, num_episodes=50):
        total_reward = 0.0
        self.previous_position = 0  # Initialize to a default value

        for _ in range(num_episodes):
            done = False

            # Get the initial game screen
            game_screen = self.get_screen_shot()

            # Preprocess the game screen
            state = preprocess_frame(game_screen)

            while not done:
                action = self.get_mouse_movement()
                # Execute the action
                self.perform_mouse_movement(action)

                # Get the next game screen
                next_game_screen = self.get_screen_shot()

                # Preprocess the next game screen
                next_state = preprocess_frame(next_game_screen)

                # Determine if the game is "done"
                current_position = self.get_player_position(state)
                done = self.is_game_done(current_position)

                reward = self.get_reward(current_position)
                self.add_experience_to_buffer(state, action, reward, next_state, done)
                self.train(BATCH_SIZE)

                # Update previous_position
                self.previous_position = current_position

                state = next_state
                total_reward += reward

        average_reward = total_reward / num_episodes
        return average_reward

    def compute_loss(self, states, actions, rewards, next_states, dones):
        # Convert data to tensors
        states = torch.FloatTensor(states).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(-1).to(device)  # Ensure that actions are a column vector
        rewards = torch.FloatTensor(rewards).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # Compute Q-values for the current states and next states
        curr_q_values = self.model(states)
        curr_q = curr_q_values.gather(1, actions).squeeze(-1)  # Gather along the action dimension
        next_q_values = self.model(next_states)
        next_q = next_q_values.max(1)[0]

        # Compute target Q-values
        target_q = rewards + (1 - dones) * self.gamma * next_q

        # Compute loss
        loss = self.loss_function(curr_q, target_q.detach())
        return loss

    def objective(self, params):
        # Unpack the parameters
        num_episodes, = params

        # Simulate the game
        average_reward = self.simulate_game(num_episodes)

        # We're minimizing the objective function, so return the negative of the reward
        return -average_reward

    def optimize(self):
        # Optimize the AI
        res = gp_minimize(self.objective, [(1, 100)], n_calls=50, random_state=0)

        # Print the result
        print(f'Best reward: {-res.fun}')

    def get_mouse_movement(self):
        if np.random.rand() < self.epsilon:
            # Perform a random action
            action = np.random.randint(OUTPUT_SIZE)
        else:
            # Get the best action according to the model
            with torch.no_grad():
                state_tensor = torch.from_numpy(self.previous_position).float().unsqueeze(0).to(device)
                q_values = self.model(state_tensor)
                action = torch.argmax(q_values).item()

        # Decay the epsilon value
        self.epsilon = max(self.epsilon * EPSILON_DECAY, EPSILON_MIN)
        return action

    def get_player_position(self, state):
        # Function to determine the player's current position
        # This depends on the specifics of your game and may need to be adjusted
        player_pixels = np.where(np.all(state == PLAYER_COLOR / 255, axis=-1))
        if player_pixels[0].size > 0:
            return np.array([player_pixels[0].mean(), player_pixels[1].mean()])
        else:
            return self.previous_position

    def is_game_done(self, current_position):
        # Function to determine if the game is "done" or not
        # This will need to be adjusted based on the specifics of your game
        done = False
        if self.previous_position is not None:
            # Consider the game done if the player has not made progress in the last 200 steps
            self.steps_since_progress += 1
            if np.any(current_position > self.previous_position):
                self.steps_since_progress = 0
            if self.steps_since_progress > 200:
                done = True
        return done

    def get_reward(self, current_position):
        # Function to calculate the reward based on the player's current position
        # This will need to be adjusted based on the specifics of your game
        if self.previous_position is not None:
            reward = np.maximum(current_position - self.previous_position, 0.0).sum()
        else:
            reward = 0.0
        return reward

    def add_experience_to_buffer(self, state, action, reward, next_state, done):
        # Adds an experience to the replay buffer
        state = torch.from_numpy(state).float().to(device)
        next_state = torch.from_numpy(next_state).float().to(device)
        self.buffer.push(state, action, reward, next_state, done)

    def predict(self, state):
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            q_values = self.model(state_tensor)
            action = torch.argmax(q_values).item()
            return action

    def train(self, batch_size):
        # Trains the AI based on experiences from the replay buffer
        if len(self.buffer) >= batch_size:
            states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
            targets = self.model(states).tolist()
            next_q_values = self.model(next_states).detach().numpy().max(axis=1)
            for i, done in enumerate(dones):
                if done:
                    targets[i][actions[i]] = rewards[i]
                else:
                    targets[i][actions[i]] = rewards[i] + next_q_values[i]
            targets = torch.tensor(targets).to(device)
            self.optimizer.zero_grad()
            loss = self.loss_function(self.model(states), targets)
            loss.backward()
            self.optimizer.step()

    def save_model(self, model_file_path='model.pth', curiosity_model_file_path='curiosity_model.pth'):
        torch.save(self.model.state_dict(), model_file_path)
        torch.save(self.curiosity_model.state_dict(), curiosity_model_file_path)
