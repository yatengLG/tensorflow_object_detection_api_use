# 支持的模型

更多tensorflow object detection 支持模型请见[git](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md)

**项目数据处理部分仅支持bbox形式数据，尚不支持mask**

|Model name|Speed (ms)|COCO mAP|Outputs|
| ---- | ---- | ---- | ---- |
|ssd_mobilenet_v1_coco | 30 | 21 | Boxes
|ssd_mobilenet_v1_0.75_depth_coco ☆ | 26 | 18 | Boxes
|ssd_mobilenet_v1_quantized_coco ☆ | 29 | 18 | Boxes
|ssd_mobilenet_v1_0.75_depth_quantized_coco ☆ |	29 |	16 |	Boxes
|ssd_mobilenet_v1_ppn_coco ☆ |	26 |	20 |	Boxes
|ssd_mobilenet_v1_fpn_coco ☆ |	56 |	32 |	Boxes
|ssd_resnet_50_fpn_coco ☆ |	76 |	35 |	Boxes
|ssd_mobilenet_v2_coco |	31 |	22 |	Boxes
|ssd_mobilenet_v2_quantized_coco |	29 |	22 |	Boxes
|ssdlite_mobilenet_v2_coco |	27 |	22 |	Boxes
|ssd_inception_v2_coco |	42 |	24 |	Boxes
|faster_rcnn_inception_v2_coco |	58 |	28 |	Boxes
|faster_rcnn_resnet50_coco |	89 |	30 |	Boxes
|faster_rcnn_resnet50_lowproposals_coco |	64 	| |	Boxes
|rfcn_resnet101_coco |	92 |	30 |	Boxes
|faster_rcnn_resnet101_coco |	106 |	32 |	Boxes
|faster_rcnn_resnet101_lowproposals_coco |	82 |	|	Boxes
|faster_rcnn_inception_resnet_v2_atrous_coco |	620 |	37 |	Boxes
|faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco |	241 |	|	Boxes
|faster_rcnn_nas 	1833 |	43 |	Boxes
|faster_rcnn_nas_lowproposals_coco |	540 |	|	Boxes


**下载的预训练好的模型会存放到此处**
