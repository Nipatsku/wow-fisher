
import pyautogui
import cv2
import imutils
import pytesseract
import time
import numpy
import math
import json

left = 749
width = 427
top = 126
height = 507
fishingRegion = (left, top, width, height)

def __main():
    time.sleep(2.0)
    print("Start.")

    while True:
        fishOnce()

    print("Finished.")

def fishOnce():
    eroseMask = create_circular_mask(5)
    print(eroseMask)

    # Throw fishing bobber.
    pyautogui.hotkey('0')

    fishing = True
    start = time.time()
    now = start
    delta = 0
    i = 0
    prevFrame = None
    prevFrameExists = False

    # Keep track of when (time s) possible bobber was detected and where (x,y)
    motionHistory = []

    while fishing and now - start <= 30.0:
        # Detect motion over fishing area.
        frame = captureFishingArea()

        if (prevFrameExists):
            diff = cv2.absdiff(prevFrame, frame)
            # grayscale + threshold
            diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            ret, diff = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            # diff = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

            # erose/dilate
            diff = cv2.erode(diff, eroseMask)
            diff = cv2.dilate(diff, eroseMask, iterations=3)

            # detect blobs
            params = cv2.SimpleBlobDetector_Params()
            params.filterByCircularity = False
            params.filterByConvexity = False
            params.filterByInertia = False
            params.filterByColor = False
            params.filterByArea = True
            params.minArea = 700
            params.maxArea = 1000000000
            params.minDistBetweenBlobs = 500
            detector = cv2.SimpleBlobDetector_create(params)
            keypoints = detector.detect(diff)
            # cache possible bobber movement.
            for keypoint in keypoints:
                motionHistory.append({
                    'frame': i,
                    'time': time.time(),
                    'location': keypoint.pt,
                    'duration': delta
                })

            # debug
            debug = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
            ret, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            debug[mask != 255] = [0, 0, 255]
            debug = cv2.drawKeypoints(debug, keypoints, numpy.array([]), (0, 255, 0),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            # Attempt detect bobber catch from motionHistory.
            # - Bobber movement lasts ~1.0 seconds.
            # - Bobber movement stays at the same location for its whole duration.

            # Remove motions that are older than memoryPeriod
            memoryPeriod = 1.2
            requiredForBobber = 0.8
            for motion in motionHistory:
                if motion['time'] < time.time() - memoryPeriod:
                    # Remove motion.
                    motionHistory.remove(motion)

            # Group motionHistory according to location.
            groupedMotionHistory = []
            for motion in motionHistory:
                location = motion['location']
                group = None
                for existingGroup in groupedMotionHistory:
                    groupLocation = existingGroup['location']
                    dist = math.sqrt(
                        math.pow(groupLocation[0] - location[0], 2) + math.pow(groupLocation[1] - location[1], 2))
                    if dist <= 100:
                        # Group together.
                        group = existingGroup
                        break
                if group is None:
                    group = {
                        'location': location,
                        'motions': []
                    }
                    groupedMotionHistory.append(group)
                motions = group['motions']
                motions.append(motion)
                # Recalculate group location by averaging its motion locations.
                locationsSum = (0, 0)
                for nMotion in motions:
                    locationsSum = (locationsSum[0] + nMotion['location'][0], locationsSum[1] + nMotion['location'][1])
                group['location'] = (int(locationsSum[0] / len(motions)), int(locationsSum[1] / len(motions)))

            # Count how long motion groups were active during the last ~second.
            for group in groupedMotionHistory:
                activeDuration = 0
                for motion in group['motions']:
                    if motion['time'] >= time.time() - memoryPeriod:
                        activeDuration += motion['duration']
                group['activeDuration'] = activeDuration

                print(json.dumps(group, indent=4))

            # Check for group that was active for long enough.
            bobberLocation = None
            for group in groupedMotionHistory:
                if group['activeDuration'] >= requiredForBobber:
                    bobberLocation = group['location']
                    break

            if bobberLocation is not None:
                # Bobber detected!
                print("Bobber detected at", bobberLocation)
                x = int(bobberLocation[0])
                y = int(bobberLocation[1])
                debug = cv2.rectangle(debug, (x - 50, y - 50), (x + 50, y + 50), (0, 255, 255), 3)

            cv2.imshow("Vision", diff)
            cv2.imshow("Debug", debug)

            if bobberLocation is not None:
                cv2.waitKey(100)
                # Move mouse to bobber location.
                xWindow = x + fishingRegion[0]
                yWindow = y + fishingRegion[1]
                delay = 0.5
                pyautogui.moveTo(xWindow, yWindow, duration=delay, tween=pyautogui.easeInOutQuad)
                # Shift + Right click
                pyautogui.keyDown('shiftleft')
                time.sleep(delay + 0.2)
                pyautogui.click()  # Extra click to focus window.
                time.sleep(0.2)
                pyautogui.rightClick()
                time.sleep(1.0)
                pyautogui.keyUp('shiftleft')

                # Clear history.
                motionHistory = []
                prevFrameExists = False
                now = time.time()

                # Exit function.
                return

        cv2.waitKey(100)
        prevFrame = frame
        prevFrameExists = True
        delta = time.time() - now
        now = time.time()
        i += 1

def create_circular_mask(size):
    mask = numpy.zeros((size, size), numpy.uint8)
    r = size / 2.0
    for x in range(0, size):
        for y in range(0, size):
            dx = (x + 0.5) - r
            dy = (y + 0.5) - r
            dist = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))
            if dist <= r:
                mask[x,y] = 1
    return mask

def captureFishingArea():
    # hardcoded region.
    screenshot = pyautogui.screenshot(region=fishingRegion)
    frame = PILtoCV(screenshot)
    return frame

def PILtoCV( pilImage ):
    openCvImage = numpy.array(pilImage)
    # Convert RGB to BGR
    openCvImage = openCvImage[:, :, ::-1].copy()
    return openCvImage

def captureBobberLabelArea():
    screenshot = pyautogui.screenshot()
    # hardcoded crop.
    left = 1620
    right = left + 152
    top = 902
    bottom = top + 31
    screenshot = screenshot.crop((left, top, right, bottom))
    #screenshot.save("screenshot.png")
    return screenshot

def checkIsBobberUnderMouse():
    bobberLabelAreaPicture = captureBobberLabelArea()
    text = pytesseract.image_to_string(bobberLabelAreaPicture)
    if "Fishing Bobber" in text:
        return True
    else:
        return False


if __name__ == '__main__':
    __main()
