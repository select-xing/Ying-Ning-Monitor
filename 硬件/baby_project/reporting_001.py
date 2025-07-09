# -*- encoding: utf-8 -*-

# Copyright (c) 2023-2024 Huawei Cloud Computing Technology Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
演示如何直接使用DeviceClient进行设备属性的上报和读写
"""
from __future__ import absolute_import
from typing import List
import time
import logging

from iot_device_sdk_python.client.client_conf import ClientConf
from iot_device_sdk_python.client.connect_auth_info import ConnectAuthInfo
from iot_device_sdk_python.iot_device import IotDevice
from iot_device_sdk_python.client.listener.property_listener import PropertyListener
from iot_device_sdk_python.client.request.service_property import ServiceProperty
from iot_device_sdk_python.client import iot_result
from core_0.yanwu import read_gpio

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(threadName)s - %(filename)s[%(funcName)s] - %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class PropertySampleListener(PropertyListener):
    def __init__(self, iot_device: IotDevice):
        """ 传入一个IotDevice实例 """
        self.device = iot_device

    def on_property_set(self, request_id: str, services: List[ServiceProperty]):
        """
        处理写属性
        :param request_id:  请求id
        :param services:    List<ServiceProperty>
        """

        """ 遍历service """
        for service_property in services:
            logger.info("on_property_set, service_id:" + service_property.service_id)
            """ 遍历属性 """
            for property_name in service_property.properties:
                logger.info("set property name:" + property_name)
                logger.info("set property value:" + str(service_property.properties[property_name]))
        self.device.get_client().respond_properties_set(request_id, iot_result.SUCCESS)

    def on_property_get(self, request_id: str, service_id: str):
        """
        处理读属性。多数场景下，用户可以直接从平台读设备影子，此接口不用实现。
        但如果需要支持从设备实时读属性，则需要实现此接口。
        :param request_id:  请求id
        :param service_id:  服务id，可选
        """
        service_property = ServiceProperty()
        service_property.service_id = "smokeDetector"
        service_property.properties = {"alarm": 10, "smokeConcentration": 36, "temperature": 64, "humidity": 32}
        services = [service_property]
        self.device.get_client().respond_properties_get(request_id, services)


def run():
    # 替换为真正的接入地址、device id和密钥，参考readme_CN.md中“上传产品模型并注册设备”
    server_uri = "cfb2991c63.st1.iotda-device.cn-north-4.myhuaweicloud.com"
    port = 8883
    device_id = "664df2467dbfd46fabbcf001_LED001"
    secret = "wyh20050130"
    # iot平台的CA证书，用于服务端校验
    iot_ca_cert_path = "/home/user/Project/huaweicloud-iot-device-sdk-python/iot_device_demo/resources/root.pem"

    connect_auth_info = ConnectAuthInfo()
    connect_auth_info.server_uri = server_uri
    connect_auth_info.port = port
    connect_auth_info.id = device_id
    connect_auth_info.secret = secret
    connect_auth_info.iot_cert_path = iot_ca_cert_path

    client_conf = ClientConf(connect_auth_info)

    device = IotDevice(client_conf)

    # 设置属性监听器
    device.get_client().set_properties_listener(PropertySampleListener(device))
    rr = 40
    hr = 80
    bt = 37
    if device.connect() != 0:
        logger.error("init failed")
        return

    # 4s后上报一次设备的属性
    time.sleep(4)
    # 按照产品模型设置属性
    service_property = ServiceProperty()
    service_property.service_id = "LEDswitchcontrol"
    service_property.properties = {"respiratory_rate": rr,"heart_rate": hr,"smoke_detection":read_gpio(),"body_temperature":bt}
    # 组装成列表的形式
    services = [service_property]

    # 上报设备属性
    device.get_client().report_properties(services)
    return device
   # while True:
       # time.sleep(5)

def reporting(device):
    """持续上报设备属性"""
    try:
        while True:
            # 动态读取GPIO最新值
            service_property = ServiceProperty()
            service_property.service_id = "LEDswitchcontrol"
            service_property.properties = {
                "respiratory_rate":rr,
                "body_temperature": bt,
                "heart_rate": hr,
                "smoke_detection": read_gpio()  # 每次循环读取最新GPIO值
            }

            device.get_client().report_properties([service_property])
            logger.info("属性上报成功")

            time.sleep(2)
    except KeyboardInterrupt:
        logger.info("停止上报")

def report_value():
    device = run()
    if device:
        reporting(device)
if __name__ == "__main__":
    device = run()  # 修改run函数返回device对象
    if device:
        reporting(device)
