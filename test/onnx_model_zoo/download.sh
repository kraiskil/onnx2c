
wget -c https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-8.tar.gz
tar xf tinyyolov2-8.tar.gz
mv tiny_yolov2/Model.onnx tiny_yolov2/model.onnx

wget -c https://github.com/onnx/models/raw/master/vision/classification/alexnet/model/bvlcalexnet-9.tar.gz
tar xf bvlcalexnet-9.tar.gz

wget -c https://github.com/onnx/models/raw/master/vision/classification/squeezenet/model/squeezenet1.1-7.tar.gz
tar xf squeezenet1.1-7.tar.gz
mv squeezenet1.1/squeezenet1.1.onnx squeezenet1.1/model.onnx

wget -c https://github.com/onnx/models/raw/master/vision/classification/squeezenet/model/squeezenet1.0-9.tar.gz
tar xf squeezenet1.0-9.tar.gz

