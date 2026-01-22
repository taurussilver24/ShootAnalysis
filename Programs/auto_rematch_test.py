import psutil
import win32gui
import win32process
import dxcam
import cv2
import pyautogui
import time
import os

# --- CONFIG ---
PROCESS_NAME = "NBA2K25.exe"
TARGET_TITLE = "NBA 2K25"  # The exact title from your log
ANCHORS = {
    'quit': 'anchor_quit.png',
    'rematch': 'anchor_rematch.png',
    'yes': 'anchor_yes.png'
}
CONFIDENCE = 0.8


def get_game_window():
    """
    Finds the specific Main Window of NBA 2K25.
    Logic matched to your successful diagnostic log.
    """
    target_pid = None
    # 1. Find PID
    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'] == PROCESS_NAME:
            target_pid = proc.info['pid']
            break

    if not target_pid: return None

    # 2. Find the Visible Window with the correct Title
    result_rect = None

    def callback(hwnd, _):
        nonlocal result_rect
        _, found_pid = win32process.GetWindowThreadProcessId(hwnd)
        if found_pid == target_pid:
            title = win32gui.GetWindowText(hwnd)
            # Filter for the Main Window (Visible + Correct Title)
            if win32gui.IsWindowVisible(hwnd) and title == TARGET_TITLE:
                result_rect = win32gui.GetWindowRect(hwnd)
                print(f"✅ Locked on Game Window: '{title}' at {result_rect}")

    win32gui.EnumWindows(callback, None)
    return result_rect


def find_and_click(camera, template_path, description, win_rect):
    if not os.path.exists(template_path):
        print(f"❌ Missing File: {template_path}")
        return False

    # 1. Grab Frame
    frame = camera.grab()
    if frame is None: return False

    # 2. Crop to Game Window
    x1, y1, x2, y2 = win_rect
    h, w, _ = frame.shape

    # Safety clamp
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1: return False

    game_frame = frame[y1:y2, x1:x2]

    # 3. Match
    img_gray = cv2.cvtColor(game_frame, cv2.COLOR_RGB2GRAY)
    template = cv2.imread(template_path, 0)

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if max_val >= CONFIDENCE:
        # Calculate Click Point
        click_x = x1 + max_loc[0] + template.shape[1] // 2
        click_y = y1 + max_loc[1] + template.shape[0] // 2

        print(f"   🎯 FOUND {description} ({max_val:.2f})! Clicking {click_x}, {click_y}...")
        pyautogui.click(click_x, click_y)
        pyautogui.moveRel(200, 0)
        return True

    return False


def main():
    print("=== NBA 2K25 Auto-Rematch (Final) ===")

    # 1. Init Camera (Monitor 0)
    try:
        camera = dxcam.create(output_idx=0, output_color="RGB")
        camera.start(target_fps=30)
    except Exception as e:
        print(f"❌ Camera Error: {e}")
        return

    print("📸 Vision System Online.")

    while True:
        # 2. Find Window (Re-check every loop to be safe)
        rect = get_game_window()

        if rect:
            # Step 1: QUIT
            if find_and_click(camera, ANCHORS['quit'], "QUIT", rect):
                time.sleep(1.5)

                # Step 2: REMATCH
                for _ in range(10):
                    if find_and_click(camera, ANCHORS['rematch'], "REMATCH", rect):
                        break
                    time.sleep(0.5)

                time.sleep(1.5)

                # Step 3: YES
                find_and_click(camera, ANCHORS['yes'], "YES", rect)
                print("✅ Restart Sequence Complete. Waiting 30s...")
                time.sleep(30)
        else:
            print("⏳ Waiting for game window...", end='\r')

        time.sleep(1)


if __name__ == "__main__":
    main()