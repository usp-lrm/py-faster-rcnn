#!/bin/bash

GPU_ID=0
PT_DIR="kitti"
NET=VGG16
WEIGHTS_DIR="output/faster_rcnn_end2end/kitti_train/"
TEST_IMDB="kitti_2012_test"
VIS=0

#NET_FINAL=vgg16_faster_rcnn_iter_56000.caffemodel
#    AP for pedestrian = 0.6025
#    AP for car = 0.8236
#    AP for cyclist = 0.6990
#    Mean AP = 0.7084

#NET_FINAL=vgg16_faster_rcnn_iter_66000.caffemodel
#    AP for pedestrian = 0.5987
#    AP for car = 0.8204
#    AP for cyclist = 0.7116
#    Mean AP = 0.7102

#NET_FINAL="vgg16_faster_rcnn_iter_70000.caffemodel"
#    AP for pedestrian = 0.6028
#    AP for car = 0.8235
#    AP for cyclist = 0.7222
#    Mean AP = 0.7161

#NET_FINAL=vgg16_faster_rcnn_iter_80000.caffemodel
#    AP for pedestrian = 0.6082
#    AP for car = 0.8210
#    AP for cyclist = 0.7038
#    Mean AP = 0.7110

NET_FINAL=vgg16_faster_rcnn_iter_88000.caffemodel
#    AP for pedestrian = 0.5993
#    AP for car = 0.8234
#    AP for cyclist = 0.7290
#    Mean AP = 0.7172


time ./tools/test_net.py \
  --gpu ${GPU_ID} \
  --def models/${PT_DIR}/${NET}/faster_rcnn_end2end/test.prototxt \
  --net ${WEIGHTS_DIR}${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  --vis ${VIS} \
  ${EXTRA_ARGS}
