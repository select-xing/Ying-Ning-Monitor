import serial


def is_valid_data(data_str):
    """验证数据有效性"""
    return len(data_str) == 28 and data_str.startswith('AA55') and data_str.endswith('0D0A')


def parse_respiratory_rate(data_str):
    """独立呼吸率解析函数"""
    if not is_valid_data(data_str):
        return None
    try:
        return int(data_str[6:8], 16)
        print(data_str[6:8])# 提取第3字节（索引6-7）
    except (ValueError, IndexError):
        return None


def parse_heart_rate(data_str):
    """独立心率解析函数"""
    if not is_valid_data(data_str):
        return None
    try:
        return int(data_str[8:10], 16)  # 提取第4字节（索引8-9）
    except (ValueError, IndexError):
        return None


def read_serial(port='/dev/ttyAMA2', baudrate=115200):
    try:
        with serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1
        ) as ser:
            print(f"成功打开串口 {ser.name}，等待数据...（按Ctrl+C退出）")

            while True:
                try:
                    raw_data = ser.readline()
                    line = raw_data.decode('utf-8').strip()
                    print(line)
                    rr = int(line[6:8],16)
                    print(rr)
                    print(line[8:10])
                    
                except UnicodeDecodeError:
                    print("警告：接收到非UTF-8数据")
                except KeyboardInterrupt:
                    print("\n用户终止操作")
                    break

    except serial.SerialException as e:
        print(f"串口错误: {str(e)}")


if __name__ == '__main__':
    read_serial()
