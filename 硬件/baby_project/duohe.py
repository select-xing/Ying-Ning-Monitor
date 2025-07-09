import multiprocessing
import psutil
import os
import time
#from yanwu import read_gpio
from reporting_001 import report_value
from core_1.pwm_gai import duoji_run
from core_1.cam_rtsp import opencv_to_ffmpeg

def core_0():
    # 绑定到第一个核心（例如核心0）
    p = psutil.Process(os.getpid())
    p.cpu_affinity([0])  # 绑定到核心0
    print(f"Core {p.cpu_affinity()}: core_0 start")
    report_value()

def core_1():
    # 绑定到第二个核心（例如核心1）
    p = psutil.Process(os.getpid())
    p.cpu_affinity([1])  # 绑定到核心1
    print(f"Core {p.cpu_affinity()}: core_1 start")
    duoji_run()



if __name__ == "__main__":
    # 创建2个进程
    p1 = multiprocessing.Process(target=core_0)
    p2 = multiprocessing.Process(target=core_1)
   # p3 = multiprocessing.Process(target=core_2)
    # 启动进程
    p1.start()
    p2.start()
   # p3.start()
    # 等待进程结束
    p1.join()
    p2.join()
  #  p3.join()
