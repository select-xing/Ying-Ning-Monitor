import time
from smbus2 import SMBus, i2c_msg  # 使用新版库


class JX90614:
    def __init__(self, bus_num=2, address=0x7F):
        self.bus = SMBus(bus_num)  # 使用新版SMBus类
        self.address = address

        # 初始化时序（保持原始C代码时序）
        self._write_register(0x30, 0x00)
        time.sleep(0.003)
        self._write_register(0x30, 0x08)
        time.sleep(0.003)

    def _write_register(self, reg, value):
        try:
            # 使用i2c_msg写入数据
            write = i2c_msg.write(self.address, [reg, value])
            self.bus.i2c_rdwr(write)
        except IOError as e:
            print(f"写入错误: {e}")

    def _read_register(self, reg):
        try:
            # 第一步：发送寄存器地址
            write = i2c_msg.write(self.address, [reg])
            # 第二步：重启总线读取数据
            read = i2c_msg.read(self.address, 1)
            self.bus.i2c_rdwr(write, read)
            return list(read)[0]
        except IOError as e:
            print(f"读取错误: {e}")
            return 0

    def read_temperature(self):
        """带错误重试的温度读取"""
        for _ in range(3):  # 添加重试机制
            try:
                adc = self._read_register(0x10)
                adc1 = self._read_register(0x11)
                adc2 = self._read_register(0x12)

                # 修正后的数据组合
                wd = (adc << 2) + (adc1 >> 6)
                wd1 = (adc << 16) | (adc1 << 8) | adc2  # 修正位移错误

                temperature = wd1 / 16384.0  # 正确计算公式
                return wd, round(temperature, 2)

            except IOError:
                time.sleep(0.01)
        return 0, 0.0


if __name__ == "__main__":
    sensor = JX90614()
    try:
        while True:
            wd, temp = sensor.read_temperature()
            print(f"组合值: 0x{wd:04X} | 温度: {temp}℃")
            time.sleep(1)
    except KeyboardInterrupt:
        sensor.bus.close()
