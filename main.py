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
    options = {"pbLoad": "model/tiny-yolo-box3.pb", "metaLoad": "model/tiny-yolo-box3.meta", "threshold": 0.1}
    tfnet = TFNet(options)
    while True:
        frame = get_capture(pipeline)
        outputs = tfnet.return_predict(frame) 
        result = predict(frame, outputs, colors, labels)
        ret, jpeg = cv2.imencode('.jpg', result)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/deploy', methods=['POST'])
def deploy():
    global tfnet
    global labels
    if request.method == 'POST':
        zip_url = request.values['zip_url']
    path = '/'.join(zip_url.split('/')[:-1])
    filename = zip_url.split('/')[-1]
    zipfile = 'darkflow/built_graph/model/'+filename
    z = zipfile.ZipFile(zipfile)
    z.extractall('model/')
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

