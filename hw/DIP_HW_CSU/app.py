from flask import Flask
from flask import render_template, request, url_for, redirect
import time
import platenumber
import cv2
import os


app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Power by Jian.Yin'


@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        f = request.files['file']
        paths = './static/imgs/' + str(time.time()) + '.jpg'
        f.save(paths)
        img_path = paths[1:]
        result = platenumber.reg(paths)

        img = cv2.imread(paths)
        h, w, c = img.shape
        print(result)
        return render_template('index.html', img_path = img_path, result=result, info = {'h': h//10, 'w': w//10})
    else:
        img_path = '/static/imgs/测试车牌1.jpg'
        result = '桂CS6736'
        img = cv2.imread('.'+img_path)
        h, w, c = img.shape
        return render_template('index.html', img_path = img_path, result=result, info = {'h':h // 10, 'w':w // 10})


if __name__ == '__main__':
    app.run(debug=True)
