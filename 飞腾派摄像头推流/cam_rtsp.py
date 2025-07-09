import cv2
import ffmpeg
import subprocess
import threading

RTSP_URL = "rtsp://localhost:8554/live/stream"

# 摄像头参数
CAMERA_ID = 0

def get_camera_resolution(camera_id=0):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise ValueError("无法打开摄像头！")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)       # 减少缓冲区
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # 优先MJPEG格式
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    return width, height, fps

def opencv_to_ffmpeg():
    cap = cv2.VideoCapture(CAMERA_ID)
    width, height, fps = get_camera_resolution(CAMERA_ID)

    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{width}x{height}', framerate=fps)
        .output(
            RTSP_URL,
            codec="libx264",  # H.264 编码
            pix_fmt="yuv420p",
            preset="ultrafast",    # 编码速度/质量平衡
            x264opts='keyint=30:min-keyint=30:no-scenecut=1',  # 固定GOP
            rtsp_transport="udp", 
            tune='zerolatency',       # 零延迟模式
            f="rtsp"
        )
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            process.stdin.write(frame.tobytes())

    finally:
        cap.release()
        process.stdin.close()
        process.wait()

if __name__ == "__main__":
    opencv_to_ffmpeg()