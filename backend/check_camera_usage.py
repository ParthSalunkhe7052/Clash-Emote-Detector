"""
Quick Camera Usage Checker
Checks if camera is being used by another application
"""

import cv2
import time

print("=" * 70)
print("CAMERA USAGE CHECKER")
print("=" * 70)
print()

print("This tool checks if your camera is available or being used by another app.")
print()

# Check camera 0
print("🔍 Checking Camera 0 (Laptop Camera)...")
try:
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)  # Use Media Foundation
    if cap.isOpened():
        time.sleep(0.5)
        ret, frame = cap.read()
        if ret:
            print("✅ Camera 0 is AVAILABLE and working!")
        else:
            print("⚠️ Camera 0 opened but cannot read frames")
            print("   This usually means another app is using it")
        cap.release()
    else:
        print("❌ Camera 0 is BUSY or not available")
        print("   Close apps like: Zoom, Skype, Teams, OBS, Discord")
except Exception as e:
    print(f"❌ Error: {e}")

print()

# Check camera 1
print("🔍 Checking Camera 1 (DroidCam)...")
try:
    cap = cv2.VideoCapture(1, cv2.CAP_MSMF)
    if cap.isOpened():
        time.sleep(0.5)
        ret, frame = cap.read()
        if ret:
            print("✅ Camera 1 is AVAILABLE and working!")
        else:
            print("⚠️ Camera 1 opened but cannot read frames")
        cap.release()
    else:
        print("❌ Camera 1 is not available")
        print("   Make sure DroidCam app is running")
except Exception as e:
    print(f"❌ Error: {e}")

print()
print("=" * 70)
print("RECOMMENDATIONS:")
print("=" * 70)
print()
print("If camera is BUSY:")
print("  1. Close Zoom, Skype, Microsoft Teams")
print("  2. Close OBS Studio, Discord video")
print("  3. Close Windows Camera app")
print("  4. Check Task Manager for camera-using apps")
print("  5. Restart your computer if needed")
print()
print("If camera is AVAILABLE:")
print("  ✅ You're good to go! Run run.bat to start the app")
print()
