import cv2
import pyrealsense2 as rs
import numpy as np
import os
import requests
import zipfile
import io
from darkflow.net.build import TFNet
from flask import Flask, render_template, Response, request
from ftplib import FTP
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
tfnet = None
labels = None
@app.route('/')
def index():
    return render_template('index.html')

def load_labels():
    labels = []
    with open('model/labels.txt') as f:
         for i in f.readlines():
             labels.append(i.replace("\n", ""))
    return labels

def get_capture(pipeline):
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    color_image = cv2.flip(color_image, 1)
    return color_image

def predict(frame, outputs, colors, labels):
    for output in outputs:
        if output['label'] not in labels: continue
        label = '{}: {:.0f}%'.format(output['label'], output['confidence'] * 100)
        cv2.rectangle(frame, (output['topleft']['x'],output['topleft']['y']), (output['bottomright']['x'],output['bottomright']['y']), colors[labels.index(output['label'])], 4)
        cv2.putText(frame, label, (output['topleft']['x'],output['topleft']['y']), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 255, 255), 4)
    return frame

def gen():
    global tfnet
    global labels
    labels = load_labels()
    pipeline = rs.pipeline()
    colors =[tuple(255 * np.random.rand(3)) for i in range(20)]
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
    pipeline.start(config)
    options = {"pbLoad": "model/model.pb", "metaLoad": "model/model.meta", "threshold": 0.4}
    tfnet = TFNet(options)
    while True:
        frame = get_capture(pipeline)
        outputs = tfnet.return_predict(frame) 
        result = predict(frame, outputs, colors, labels)
        ret, jpeg = cv2.imencode('.jpg', result)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/deploy')
def deploy():
    global tfnet
    global labels
    if 'url' in request.args:
        zip_url = request.args['url']
    ftp = FTP('10.0.0.95')
    ftp.login('dl-ftp','123')
    path = '/'.join(zip_url.split('/')[:-1])
    filename = zip_url.split('/')[-1]
    ftp.cwd(path)
    with open('model/'+filename, 'wb') as localfile:
        ftp.retrbinary('RETR ' + filename, localfile.write, 1024)
        ftp.quit()
    if os.path.isfile('model/'+filename): 
        localfile = open('model/'+filename, 'rb')
        z = zipfile.ZipFile(localfile)
        z.extractall('model/')
        os.remove('model/'+filename) 
#    if zip_url == '1':
#         copyfile('built_graph/tiny-yolo-2c.meta', 'model/model.meta')
#         copyfile('built_graph/tiny-yolo-2c.pb', 'model/model.pb')
#         copyfile('built_graph/labels_2.txt', 'model/labels.txt')
#    elif zip_url == '2':
#         copyfile('built_graph/tiny-yolov2-4c.meta', 'model/model.meta')
#         copyfile('built_graph/tiny-yolov2-4c.pb', 'model/model.pb')
#         copyfile('built_graph/labels_4.txt', 'model/labels.txt')
    options = {"pbLoad": "model/model.pb", "metaLoad": "model/model.meta", "threshold": 0.4}
    tfnet = TFNet(options)
    labels = load_labels()
    return Response('sucess', status=200)

@app.route('/video_feed')
def video_feed():
  return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

