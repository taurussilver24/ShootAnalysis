import json
import os

STATUS_FILE = "session_status.json"

def write_status(state):
    """
    States:
    'PLAYING' - Game is live, Detector is recording.
    'FINISHED' - Game Ended (Buzzer/Scoreboard detected).
    'RESTARTING' - Bot is currently clicking buttons.
    """
    with open(STATUS_FILE, 'w') as f:
        json.dump({"state": state}, f)

def get_status():
    if not os.path.exists(STATUS_FILE):
        return "PLAYING" # Default
    try:
        with open(STATUS_FILE, 'r') as f:
            data = json.load(f)
        return data.get("state", "PLAYING")
    except:
        return "PLAYING"