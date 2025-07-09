import json
import os
from flask import Flask, redirect, render_template, url_for, request, make_response, jsonify, Response
from flask_cors import CORS
from werkzeug.routing import BaseConverter
import pymysql
import re
import random
import time
from yolov5_dnn import detectimg
import cv2
from datetime import datetime
from cry_detect import cryDetecting
import numpy as np

host = '192.168.137.1'
port = 5000

# 创建路由, __name__代表使用当前文件的根目录作为路由.
# static_url_path是静态资源的url前缀
# static_folder是静态资源的文件夹
# tamplate_folder默认参数是templates, 模板文件的目录
app = Flask(__name__)
CORS(app)

conn = pymysql.connect(host='127.0.0.1', user='root', password='Tlsldesg151213,,', charset='utf8mb4')
cursor = conn.cursor()
cursor.execute('use wlwdatabase')
cursor.execute('COMMIT')

direction_status = {
    'left':False,
    'right':False,
    'up':False,
    'down':False
}

# 使用配置文件
# app里面有config对象, 用于使用配置文件
# app.config.from_pyfile('config.cfg')
# 也可以自定义类对象, 从对象中获取
# 在config后面跟[]也可以直接操作字典对象

# 想要自定义转换器, 必须重新实现转换器类的方法
# 这个类不是flask下的, 而是他基于的环境werkzeug.routing里面的BaseConverter
class RegexConverter(BaseConverter):
    # 转换器类在构造的时候会自动接收url_map列表, 是固定的, 自己传入的正则表达式参数要写在后面
    def __init__(self, url_map, regex):
        # 调用父类方法, 把url_map传入
        super().__init__(url_map)
        # 在父类里面是有regex的, 默认是匹配不为/的所有字符, 这里改成了自己传入的参数
        # regex = "[^/]+"
        self.regex = regex
    pass
# 但是重新实现了类之后, flask也是找不到这个类的, 要加入到转换器对应的字典里面, 自己起一个名字
# 比如自带的转换器他起的名字就是int, float等等
app.url_map.converters['re'] = RegexConverter

controlDirection = ''


@app.route('/',methods=['GET'])
# route可以跟methods参数, 接收列表, 设置可以访问的方式
def index():
    return 'hello world'

# 这里是使用自带的转换器int, 然后通过函数接收转换器的参数
@app.route("/goods/<re(r'\d+'):id>")
def goods(id):
    return 'the id is %s'%id

# 自定义响应信息
@app.route('/res',methods=['GET','POST'])
def res():
    # 直接返回的话依次是响应体, 状态码, 响应头
    # 响应头可以是列表里面带元组, 也可以是字典
    # return r'\0success', 200, {'city': 'shenzhen', 'name': 'zhangsan'}
    # 也可以构建响应信息的对象
    # 在创建对象时候传入的就是响应体, 然后状态码和响应头可以自己设置
    resp = make_response('response page')
    resp.status = 200
    resp.headers['city'] = 'cz'
    return resp

@app.route('/getjson',methods=['GET','POST'])
def getjson():
    # 使用jsonify函数可以将字典转换成json字符串的同时, 设置响应体类型为json格式
    data = {
        'city':'shenzhen',
        'name':'zhangsan'
    }
    return jsonify(data)

# flask中前端往后端发送数据的时候, 会被保存在全局对象request里面, 然后可以导入request去拿
@app.route('/postdata',methods=['POST'])
def postdata():
    name = request.form.get('name')
    age = request.form.get('age')
    print(request.data)
    data = {
        'name':name,
        'age':age
    }
    return jsonify(data)

@app.route('/login',methods=['GET','POST'])
def login():
    if not request.data:
        return '-1'
    data = json.loads(request.data)
    username = data.get('username')
    password = data.get('password')
    print(data)
    if username and password:
        # print(username,password)
        cursor.execute('select password from usertable where username="%s"'%username)
        real_password = cursor.fetchone()
        cursor.execute('COMMIT')
        if real_password and password == str(real_password[0]):
            return '1'
    return '-1'

@app.route('/register',methods=['GET','POST'])
def register():
    if not request.data:
        return '-1'
    data = json.loads(request.data)
    username = data.get('username')
    password = data.get('password')
    print(data)
    if username and password:
        # print(username,password)
        cursor.execute('select * from usertable where username="%s"' % username)
        result = cursor.fetchone()
        print(result)
        if result:
            return '-1'
        else:
            cursor.execute(f"INSERT INTO usertable (username, password) VALUES ('{username}', '{password}')")
            cursor.execute('COMMIT')
        return '1'
    return '-1'

@app.route('/uploadjpg',methods=['POST'])
def uploadjpg():
    data = request.data
    with open('./savejpg.jpeg','wb') as fp:
        fp.write(data)
    return '1'

@app.route('/downloadjpg',methods=['GET'])
def downloadjpg():
    if os.path.exists('./savejpg.jpeg'):
        data = open('./savejpg.jpeg','rb').read()
        res = make_response()
        res.data = data
        res.status_code = 200
        res.mimetype = 'image/jpeg'
        print(data)
        return res
def gen(detect:bool = False):
    while True:
        if detect:
            frame = cv2.imread('./savejpg.jpeg')
            frame = detectimg('./best.onnx',frame)
            ret, bframe = cv2.imencode('.jpg',frame)
            res = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + bframe.tobytes() + b'\r\n'
            yield res
            time.sleep(0.1)
        else:
            with open('./savejpg.jpeg', 'rb') as fp:
                frame = fp.read()
                res = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
                yield res
                time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    return Response(gen(False),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_det')
def video_feed_det():
    return Response(gen(True),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/downloadpng',methods=['GET'])
def downloadpng():
    if not request.data:
        return '-1'
    data = json.loads(request.data)
    print(data)
    name = data.get('name')
    if name and os.path.exists('./photos/'+name+'.png'):
        with open('./photos/'+name+'.png','rb') as fp:
            data = fp.read()
            res = make_response()
            res.data = data
            res.status_code = 200
            res.mimetype = 'image/png'
            return res
    return '-1'

@app.route('/getpng/<string:name>',methods=['GET'])
def getpng(name):
    print(name)
    _, ext = os.path.splitext(name)
    if name and os.path.exists('./photos/'+name):
        with open('./photos/'+name,'rb') as fp:
            data = fp.read()
            res = make_response()
            res.data = data
            res.status_code = 200
            res.mimetype = 'image/' + ext[1:]
            return res
    return '-1'

@app.route('/getshopdata',methods=['GET'])
def getshopdata():
    retdata = []
    directory_path = './photos'
    text_dict = dict()
    name_dict = dict()
    url_dict = dict()
    with open(directory_path + '/text.txt', 'r', encoding='utf-8') as fp:
        for line in fp:
            try:
                first_part, second_part = line.split(" ", 1)
                if second_part:
                    second_part = re.sub('\n','',second_part)
                else:
                    second_part = "无描述"
            except ValueError:
                # 如果没有空格，则first_part是原字符串，second_part为空字符串或其他默认值
                if line[-1] == '\n':
                    first_part = line[:-1]
                else:
                    first_part = line
                second_part = "无描述"
            text_dict[first_part] = second_part
    with open(directory_path + '/name.txt', 'r', encoding='utf-8') as fp:
        for line in fp:
            try:
                first_part, second_part = line.split(" ", 1)
                if second_part:
                    second_part = re.sub('\n', '', second_part)
                else:
                    second_part = "无名称"
            except ValueError:
                # 如果没有空格，则first_part是原字符串，second_part为空字符串或其他默认值
                if line[-1] == '\n':
                    first_part = line[:-1]
                else:
                    first_part = line
                second_part = "无名称"
            name_dict[first_part] = second_part
    with open(directory_path + '/urls.txt', 'r', encoding='utf-8') as fp:
        for line in fp:
            try:
                first_part, second_part = line.split(" ", 1)
                if second_part:
                    second_part = re.sub('\n', '', second_part)
                else:
                    second_part = "https://www.4399.com/"
            except ValueError:
                # 如果没有空格，则first_part是原字符串，second_part为空字符串或其他默认值
                if line[-1] == '\n':
                    first_part = line[:-1]
                else:
                    first_part = line
                second_part = "https://www.4399.com/"
            url_dict[first_part] = second_part
    # print(text_dict)
    # print(name_dict)
    # print(url_dict)
    for filename in os.listdir(directory_path):
        name, ext = os.path.splitext(filename)
        if text_dict.get(name) and name_dict.get(name):
            # retdata += '{"name":"' + name_dict[name] + '","image":' + f'"http://{host}:{port}/getpng/{filename}"'
            # retdata += f',"text":"{text_dict[name]}"'
            # retdata += f',"url":"{url_dict[name]}"' + '},'
            retdata.append({'name':name_dict[name],
                            'image':f'http://{host}:{port}/getpng/{filename}',
                            'text':text_dict[name],
                            'url':url_dict[name]
                            })
    # print(retdata)
    ret_json = json.dumps(retdata,ensure_ascii=False)
    return ret_json

@app.route('/getlunbotu',methods=['GET'])
def getlunbotu():
    filenames = ['yinger1.jpg','yinger2.jpg','yinger3.jpg']
    retdata = '['
    for filename in filenames:
        name, ext = os.path.splitext(filename)
        retdata += '{"name":"' + name + '","image":' + f'"http://{host}:{port}/getpng/{filename}"' + '},'
    retdata = retdata[:-1]
    retdata += ']'
    return retdata

@app.route('/xiaotieshi')
def xiaotieshi():
    text_lst = []
    with open('./tieshi.txt', 'r', encoding='utf-8') as fp:
        for line in fp:
            text_lst.append(line)
    return random.choice(text_lst)

@app.route('/com/upload', methods=['POST'])
def com_upload():
    if not request.data:
        return '-1'
    data = json.loads(request.data)
    user = data.get('user')
    text = data.get('text')
    # print(data)
    if user and text:
        sql = "INSERT INTO comdatatable (user, text, time) VALUES (%s, %s, %s)"
        now = datetime.now()
        cursor.execute(sql, (user, text, now))
        cursor.execute('COMMIT')
        return '1'
    return '-1'

@app.route('/com/show')
def com_show():
    try:
        sql = 'select user, text, time from comdatatable order by time DESC'
        cursor.execute(sql)
        data = cursor.fetchall()
        ret_data = []
        for item in data:
            ret_data.append({'user': item[0], 'text': item[1], 'time': str(item[2])})
        ret_json = json.dumps(ret_data, ensure_ascii=False)
        # print(ret_json)
        return ret_json
    except Exception as e:
        print(e)
        return "{}"
    

@app.route('/cry_detect')
def cry_detect():
    result = cryDetecting()
    return result

@app.route('/player')
def player():
    return render_template('player.html')

@app.route('/player/handler', methods=['POST'])
def handler():
    data = request.get_json()
    print(data)
    return data

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if file.filename == '':
        return "Empty filename", 400
    
    # 直接从内存读取图片数据
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        return "Invalid image file", 400
    
    # 显示图片（窗口会自动创建）
    cv2.imshow('Uploaded Image', img)
    cv2.waitKey(0)  # 显示3秒
    cv2.destroyAllWindows()
    
    return "Image displayed successfully", 200

@app.route('/control', methods=['POST'])
def control():
    print(request.get_json())
    data = request.get_json()
    for key, value in data.items():
        direction_status[key] = value
    return jsonify({"status": "success", "message": "Control command received"})

@app.route('/get/control_code', methods=['GET'])
def getControlCode():
    for key, value in direction_status.items():
        if value == True:
            if key == 'left':
                return 'a'
            elif key == 'right':
                return 'd'
            elif key == 'up':
                return 'w'
            elif key == 'down':
                return 's'
    return 'n'

if __name__ == '__main__':
    app.run(host=host,port=port)
