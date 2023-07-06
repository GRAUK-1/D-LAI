# Game details
GAME_TITLE = 'Getting Over It'

# Hyperparameters
OUTPUT_SIZE = 2
BATCH_SIZE = 64
EPSILON = 0.9
EPSILON_DECAY = 0.999
EPSILON_MIN = 0.1
BUFFER_MAXLEN = int(1e5)
REWARD_THRESHOLD = 100
SAVE_INTERVAL = 600  # Save model every 10 minutes

# Miscellaneous
MODEL_SAVE_PATH = 'model_save.pth'
