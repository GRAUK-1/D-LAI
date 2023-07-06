import cv2
import pyautogui
import time

# Define the number of screenshots to capture
num_screenshots = 1000

# Define the directory to save the screenshots
screenshot_dir = 'screenshots/'

# Play the game and capture screenshots
for i in range(num_screenshots):
    # Capture a screenshot
    screenshot = pyautogui.screenshot()

    # Save the screenshot
    screenshot.save(screenshot_dir + 'screenshot' + str(i) + '.png')

    # Wait a short time before capturing the next screenshot
    time.sleep(0.1)
