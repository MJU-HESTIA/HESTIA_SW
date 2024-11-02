import cv2
import RPi.GPIO as GPIO
import time

# 서보 모터 핀 설정 (좌우 제어만 15번 핀 사용)
pan_pin = 15  # 좌우(Pan) 제어 서보모터 핀

# 서보 모터 제어 초기화
GPIO.setmode(GPIO.BOARD)
GPIO.setup(pan_pin, GPIO.OUT)

# 서보모터 PWM 객체 생성 (PWM 주파수 50Hz)
pan_servo = GPIO.PWM(pan_pin, 50)

# 초기 서보 모터 위치 (90도가 정면을 바라보도록 설정)
pan_angle = 90  # 좌우 서보모터 초기 각도

# 서보 모터를 초기 위치로 설정 (90도: 정면)
pan_servo.start(pan_angle / 18 + 2)
time.sleep(1)  # 서보 모터가 초기 위치로 이동할 시간을 줌

# 얼굴 인식용 Haar Cascade 로드
face_cascade = cv2.CascadeClassifier("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")

def set_servo_angle(servo, angle):
    """서보 모터 각도를 설정하는 함수 (각도를 0~180 사이로 제한)"""
    angle = max(0, min(180, angle))  # 각도를 0과 180 사이로 제한
    duty = angle / 18 + 2
    servo.ChangeDutyCycle(duty)
    print(f"Servo angle set to: {angle}")  # 디버깅을 위해 각도 출력
    time.sleep(0.05)  # 모터가 움직일 시간을 확보

def track_face(frame, pan_angle):
    """얼굴을 감지하고 좌우 서보 모터를 제어하는 함수"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]  # 첫 번째 얼굴만 추적
        face_center_x = x + w // 2

        frame_center_x = frame.shape[1] // 2

        # 좌우(Pan) 제어
        if face_center_x < frame_center_x - 50:  # 얼굴이 화면 왼쪽에 있을 때
            pan_angle += 2  # 오른쪽으로 서보 모터 이동
            pan_angle = max(0, min(180, pan_angle))  # 각도를 0과 180 사이로 제한
        elif face_center_x > frame_center_x + 50:  # 얼굴이 화면 오른쪽에 있을 때
            pan_angle -= 2  # 왼쪽으로 서보 모터 이동
            pan_angle = max(0, min(180, pan_angle))  # 각도를 0과 180 사이로 제한

        # 서보 모터에 새 각도 적용
        set_servo_angle(pan_servo, pan_angle)

        # 얼굴을 감지한 영역에 사각형을 그리기
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return pan_angle

def gstreamer_pipeline(
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=True"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def show_camera():
    window_title = "Face Tracking (Pan Only)"
    video_capture = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

    if video_capture.isOpened():
        global pan_angle

        try:
            cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    break

                # 얼굴 추적 및 좌우 서보 모터 제어
                pan_angle = track_face(frame, pan_angle)

                # 얼굴 인식 결과와 함께 화면에 표시
                cv2.imshow(window_title, frame)

                keyCode = cv2.waitKey(10) & 0xFF
                if keyCode == 27 or keyCode == ord('q'):
                    break
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Unable to open camera")


if __name__ == "__main__":
    try:
        show_camera()
    finally:
        # 서보 모터 정지 및 GPIO 정리
        try:
            pan_servo.stop()
        except:
            pass
        GPIO.cleanup()
