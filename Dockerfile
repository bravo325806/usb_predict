FROM ubuntu:16.04
MAINTAINER bravo325806 bravo325806@gmail.com

WORKDIR /
RUN apt-get update
RUN apt-get install python3-pip git libglib2.0-0 libsm6 libxrender1 -y
RUN pip3 install tensorflow opencv-python flask flask-cors
RUN git clone https://github.com/bravo325806/rtmp_predict
WORKDIR /rtmp_predict

CMD python3 main.py
