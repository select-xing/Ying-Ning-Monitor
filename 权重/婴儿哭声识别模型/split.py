from pydub import AudioSegment
import os


def split_audio(input_folder, output_folder, split_num=2):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹内的所有wav文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            file_path = os.path.join(input_folder, filename)

            try:
                # 载入音频文件
                audio = AudioSegment.from_wav(file_path)
                duration = len(audio)  # 音频总时长（毫秒）

                # 计算分割点
                split_point = duration // split_num

                # 分割并保存
                base_name = os.path.splitext(filename)[0]
                for i in range(split_num):
                    start = i * split_point
                    end = (i + 1) * split_point if i < split_num - 1 else duration
                    segment = audio[start:end]

                    output_path = os.path.join(
                        output_folder,
                        f"{base_name}_{i}.wav"
                    )
                    segment.export(output_path, format="wav")
                    print(f"已保存：{output_path}")

            except Exception as e:
                print(f"处理文件 {filename} 时出错：{str(e)}")


# 使用示例
input_dir = "D:/pycharmproject/baby_cry_classify/baby_cry_data/split_train/sleepy"  # 替换为输入文件夹路径
output_dir = "D:/pycharmproject/baby_cry_classify/baby_cry_data/split_train/split_sleepy"  # 替换为输出文件夹路径
split_audio(input_dir, output_dir)