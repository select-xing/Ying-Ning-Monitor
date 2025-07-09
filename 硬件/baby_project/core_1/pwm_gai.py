import time
#import sys
#import readchar
from smbus2 import SMBus
import requests as rq

class PCA9685:
    def __init__(self, bus=2, address=0x40):
        self.bus = SMBus(bus)
        self.address = address
        self._initialize()
    
    def _initialize(self):
        # 初始化芯片
        self._write_byte(0x00, 0x20)  # MODE1: 自动增量使能
        time.sleep(0.05)
        self.set_pwm_freq(50)         # 舵机标准频率50Hz
    
    def _write_byte(self, reg, value):
        self.bus.write_byte_data(self.address, reg, value)
    
    def _write_word(self, reg, value):
        self.bus.write_word_data(self.address, reg, value)
    
    def set_pwm_freq(self, freq):
        # 设置PWM频率（24-1526 Hz）
        prescale = int(25000000.0 / (4096 * freq) - 1)
        self._write_byte(0x00, 0x10)   # 进入睡眠模式
        self._write_byte(0xFE, prescale)
        self._write_byte(0x00, 0x20)   # 退出睡眠模式
        time.sleep(0.005)
    
    def set_servo_180angle(self, channel, angle): #180du duoji
        # 角度转PWM（0-180度 -> 0.5ms-2.5ms）
        pulse_width = 500 + (2000 * angle / 180)  # 单位微秒
        count = int(pulse_width * 4096 / 20000)   # 20ms周期
        
        reg = 0x06 + 4 * channel  # LED0_ON_L寄存器
        # 设置ON时间（固定为0）
        self._write_byte(reg, 0x00)
        self._write_byte(reg + 1, 0x00)
        # 设置OFF时间
        self._write_byte(reg + 2, count & 0xFF)
        self._write_byte(reg + 3, (count >> 8) & 0x0F)

    def set_servo_270angle(self, channel, angle): #270du duoji
        # 角度转PWM（0-270度 -> 0.5ms-2.5ms）
        pulse_width = 500 + (2000 * angle / 270)  # 单位微秒
        count = int(pulse_width * 4096 / 20000)   # 20ms周期
        
        reg = 0x06 + 4 * channel  # LED0_ON_L寄存器
        # 设置ON时间（固定为0）
        self._write_byte(reg, 0x00)
        self._write_byte(reg + 1, 0x00)
        # 设置OFF时间
        self._write_byte(reg + 2, count & 0xFF)
        self._write_byte(reg + 3, (count >> 8) & 0x0F)
def duoji_run():
    pca = PCA9685(bus=2)  # 指定使用I2C总线2
    angle = 135
    angle_1 = 20
    url = "http://192.168.13.84:5000/get/control_code"
    try:
        while True:
            # 示例：控制通道0的舵机往复运动
            key_lower = rq.get(url).text
            # key_lower = readchar.readkey().lower()
            if key_lower == 'a':
                if 0 < angle and angle < 270:
                    angle = angle + 5
                    print("a")
                    pca.set_servo_270angle(0, angle)
            elif key_lower == 'd':
                if 5 < angle and angle < 270:
                    angle = angle - 5
                    print("d")
                    pca.set_servo_270angle(0, angle)
            elif key_lower == 'w':
                if 0 < angle_1 and angle_1 < 180:
                    angle_1 = angle_1 + 5
                    print("w")
                    pca.set_servo_180angle(1,angle_1)
            elif key_lower == 's':
                if 5 < angle_1 and angle_1 < 180:
                    angle_1 = angle_1 - 5
                    print("s")
                    pca.set_servo_180angle(1,angle_1)           
                #pca.set_servo_180angle(1,angle_1)

                
    except KeyboardInterrupt:
        print("\n程序终止")    
if __name__ == "__main__":
    duoji_run()
