FROM ubuntu:16.04

WORKDIR /

RUN apt-get update && apt-get install libusb-1.0-0-dev vim git libsm6 libxext6 libxrender-dev libgtk2.0-dev  python3.5 python3-pip -y
RUN git clone https://github.com/bravo325806/usb_predict

WORKDIR /usb_predict

RUN pip3 install pyrealsense2 opencv-python tensorflow flask flask-cors numpy==1.16.4 requests

CMD python3 main.py
