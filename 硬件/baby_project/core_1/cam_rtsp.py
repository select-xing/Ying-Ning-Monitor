import cv2
import ffmpeg
import subprocess
import threading

# RTSP 服务器地址（MediaMTX 默认端口：8554）
RTSP_URL = "rtsp://192.168.13.84:8554/live/stream"

# 摄像头参数（根据实际情况调整）
CAMERA_ID = 1  # 默认摄像头（或 DirectShow 设备名称，如 "Integrated Camera"）

def get_camera_resolution(camera_id=0):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise ValueError("无法打开摄像头！")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)       # 减少缓冲区
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # 优先MJPEG格式
    # 获取摄像头实际分辨率（可能因驱动不同返回值有差异）
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    return width, height, fps

def opencv_to_ffmpeg():
    # 1. 用 OpenCV 打开摄像头
    cap = cv2.VideoCapture(CAMERA_ID)
    width, height, fps = get_camera_resolution(CAMERA_ID)

    # 2. 启动 FFmpeg 进程（通过管道推流）
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{width}x{height}', framerate=fps)
        .output(
            RTSP_URL,
            codec="libx264",  # H.264 编码
            pix_fmt="yuv420p",
            preset="ultrafast",    # 编码速度/质量平衡
            x264opts='keyint=30:min-keyint=30:no-scenecut=1',  # 固定GOP
            rtsp_transport="udp",  # 更稳定的 TCP 传输
            tune='zerolatency',       # 零延迟模式
            max_delay='100000',    # 最大延迟
            f="rtsp"
        )
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    # 3. 循环读取帧并推流
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # 将 OpenCV 的 BGR 帧写入 FFmpeg 管道
            process.stdin.write(frame.tobytes())

    finally:
        cap.release()
        process.stdin.close()
        process.wait()

if __name__ == "__main__":
    opencv_to_ffmpeg()
