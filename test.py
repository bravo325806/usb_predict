
test = 'ftp://10.0.0.95/machan-group1-painting-yolo/tiny-yolov2-2c.zip'
path = '/'.join(test.split('/')[:-1])
file = test.split('/')[-1]
print(path, file)
