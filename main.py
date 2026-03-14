import cv2
import numpy as np
import time
import spidev
from adafruit_servokit import ServoKit

# Initialize PCA9685
kit = ServoKit(channels=16)

# Set initial positions (servos 0-3: 140Â°, servo4: 40Â°, servo5: 0Â°)
for i in range(4):
    kit.servo[i].angle = 140
kit.servo[4].angle = 40  # Servo5
kit.servo[5].angle = 180   # Wrist servo on channel 6

# Initialize SPI for MCP3008 (FSR sensor)
spi = spidev.SpiDev()
spi.open(0, 0)  # SPI bus 0, device 0
spi.max_speed_hz = 1350000

def read_adc(channel):
    if channel < 0 or channel > 7:
        return -1
    adc = spi.xfer2([1, (8 + channel) << 4, 0])
    return ((adc[1] & 3) << 8) + adc[2]

# Load object detection model
classNames = []
classFile = "/home/group10/project/Object_Detection_Files/coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "/home/group10/project/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/group10/project/Object_Detection_Files/frozen_inference_graph.pb"
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    objectInfo = []
    if len(objects) == 0:
        objects = classNames
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, className.upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    return img, objectInfo

def calculate_edge_distance(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    left1, right1, top1, bottom1 = x1, x1 + w1, y1, y1 + h1
    left2, right2, top2, bottom2 = x2, x2 + w2, y2, y2 + h2
    if (right1 < left2 or right2 < left1 or bottom1 < top2 or bottom2 < top1):
        dx = max(left1 - right2, left2 - right1, 0)
        dy = max(top1 - bottom2, top2 - bottom1, 0)
        return np.sqrt(dx**2 + dy**2)
    else:
        return 0

def move_servos_simultaneously(do_wrist_movement=False):
    steps = 95
    for step in range(steps):
        # Servos 0-3: 140 -> 40
        angle_0_3 = 140 - step
        for i in range(4):
            kit.servo[i].angle = angle_0_3
        
        # Servo4: 40 -> 140
        angle_4 = 40 + step
        kit.servo[4].angle = angle_4
        
        # Read FSR during movement
        fsr_value = read_adc(0)
        print(f"FSR during movement: {fsr_value}")
        if fsr_value > 50:
            if do_wrist_movement:
                # Wait 1 second before moving wrist
                time.sleep(1.0)
                
                # Move wrist servo (channel 6) to 90Â°
                for angle in range(150, 180):
                    kit.servo[5].angle = angle
                    time.sleep(0.02)
            
            return True
        
        time.sleep(0.05)
    return False

def reset_servos():
    target_angles = [140, 140, 140, 140, 40, 180]  # Final positions for each servo
    step_delay = 0.05  # Adjust the speed of movement (increase for slower motion)

    for step in range(10):  # Number of steps to smooth the transition
        for i in range(4):  # Fingers
            kit.servo[i].angle += (target_angles[i] - kit.servo[i].angle) / (10 - step)
        kit.servo[4].angle += (target_angles[4] - kit.servo[4].angle) / (10 - step)
        kit.servo[5].angle += (target_angles[5] - kit.servo[5].angle) / (10 - step)
        time.sleep(step_delay)

    # Ensure they reach the exact target positions
    for i in range(4):
        kit.servo[i].angle = 140
    kit.servo[4].angle = 40  
    kit.servo[5].angle = 180


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 10)

    while True:
        distance_zero_detected = False
        delay_start_time = None
        processing_active = True
        object_type = None  # To track whether we're handling apple or bottle

        try:
            while processing_active:
                success, img = cap.read()
                if not success:
                    break

                result, objectInfo = getObjects(img, 0.45, 0.2, objects=['bottle', 'apple', 'person'])
                
                person_box = None
                object_box = None
                object_type = None

                for obj in objectInfo:
                    box, className = obj
                    if className == "person":
                        person_box = box
                    elif className in ['bottle', 'apple']:
                        object_box = box
                        object_type = className

                if person_box is not None and object_box is not None:
                    distance = calculate_edge_distance(person_box, object_box)
                    cv2.putText(img, f"Distance: {int(distance)}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    print(f"Distance: {int(distance)} pixels | Object: {object_type}")

                    if distance == 0:
                        if not distance_zero_detected:
                            distance_zero_detected = True
                            delay_start_time = time.time()
                        elif time.time() - delay_start_time >= 0.3:
                            # Only do wrist movement for apples, not bottles
                            do_wrist = (object_type == 'apple')
                            if move_servos_simultaneously(do_wrist_movement=do_wrist):
                                print(f"FSR threshold reached - {'with' if do_wrist else 'without'} wrist movement")
                                
                                # Wait for 10 seconds
                                print("Waiting for 10 seconds...")
                                time.sleep(7)
                                
                                # Reset flags and continue processing
                                processing_active = False
                                distance_zero_detected = False
                                
                                # Wait for person to be detected again
                                redetection_time = time.time()
                                while True:
                                    success, img = cap.read()
                                    if not success:
                                        break
                                    
                                    result, objectInfo = getObjects(img, 0.45, 0.2, objects=['person'])
                                    person_detected = any(obj[1] == "person" for obj in objectInfo)
                                    
                                    if person_detected:
                                        if time.time() - redetection_time >= 1:
                                            # Reset all servos to default positions
                                            reset_servos()
                                            break
                                    else:
                                        redetection_time = time.time()
                                    
                                    cv2.imshow("Output", img)
                                    if cv2.waitKey(1) & 0xFF == ord('q'):
                                        raise KeyboardInterrupt
                                
                                break
                    else:
                        distance_zero_detected = False

                cv2.imshow("Output", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt

        except KeyboardInterrupt:
            print("Program stopped")
            break

    cap.release()
    cv2.destroyAllWindows()
    spi.close()s
