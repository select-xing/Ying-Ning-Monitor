import os
import time

#此代码用于读取烟雾传感器输出
def read_gpio():
    # 初始化GPIO
    if not os.path.exists("/sys/class/gpio/gpio474"):
        os.system("echo 474 > /sys/class/gpio/export")
        time.sleep(0.1)  # 等待GPIO导出完成

    # 读取GPIO值
    with open("/sys/class/gpio/gpio474/value", "r") as f:
        value = f.read().strip()
        value = 1 - int(value)
    return value


# 使用示例
if __name__ == "__main__":
    try:
        while True:
            print(f"当前GPIO值: {read_gpio()}")
            time.sleep(1)
    except KeyboardInterrupt:
        os.system("echo 474 > /sys/class/gpio/unexport")
        print("\nGPIO已清理")
