import dxcam
import cv2
import time


def check_monitors():
    print("=== DXCAM MONITOR DETECTIVE ===")
    print("Press any key to move to the next monitor check...")
    print("------------------------------------------------")

    # Check indices 0 to 4 (Most setups don't have more than 4)
    for i in range(5):
        try:
            # Try to initialize camera on this index
            camera = dxcam.create(output_idx=i, output_color="BGR")

            # If successful, grab a frame
            print(f"\n✅ Monitor Index [{i}] DETECTED!")
            print(f"   -> Resolution: {camera.width}x{camera.height}")
            print(f"   -> Rotation: {camera.rotation}")

            camera.start(target_fps=1)
            frame = camera.get_latest_frame()

            if frame is not None:
                # Add a massive label to the image
                label = f"MONITOR INDEX: {i}"
                cv2.putText(frame, label, (100, 300), cv2.FONT_HERSHEY_SIMPLEX,
                            3, (0, 0, 255), 5)

                # Show window
                window_name = f"Identify Monitor {i}"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow(window_name, frame)

                print(f"   👀 Look at your screens! A window named '{window_name}' should appear.")
                print("   (Press any key in the window to continue)")

                cv2.waitKey(0)  # Wait for you to press a key
                cv2.destroyWindow(window_name)

            camera.stop()
            del camera

        except Exception as e:
            # If dxcam fails, it usually means the index doesn't exist
            if "Index out of range" in str(e) or "Failed to find output" in str(e):
                print(f"\n❌ Monitor Index [{i}] does not exist. Stopping scan.")
                break
            else:
                print(f"\n⚠️ Error checking Index [{i}]: {e}")

    print("\n=== SCAN COMPLETE ===")


if __name__ == "__main__":
    check_monitors()