项目名称：baby_project_硬件代码
部署平台：教育派
运行系统：loongnix
运用核心：core_0，core_1
主要功能：
1.core_0：读取婴儿的呼吸率、心率、体温和检测吸烟。此功能源代码位置为./core_0，分别对应leida__001.py，wd.py，yanwu.py。
同时还可以将数据上传到华为云，此功能的源代码为./reporting_001.py。
读取婴儿的哭声音频数据，并对其做出分类。此功能源代码位置为./core_0。
2.core_1：发送视频流及控制摄像头运动。此功能源代码位置为./core_1
可执行文件位置：./dtsi/duohe
备注：本项目使用了华为云的sdk
下载指令：git clone https://github.com/huaweicloud/huaweicloud-iot-device-sdk-python
