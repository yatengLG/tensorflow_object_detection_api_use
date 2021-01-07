# tensorflow objectdetection api便捷训练


|参数名|说明|备注|
|----|----|----|
|model_name|模型名|可以查看pretrained_models/model_zoo查看支持的模型|
|train_record|训练数据TFrecord|与eval_record、label_pbtxt一起提供，将该数据直接用于训练|
|eval_record|测试数据TFrecord| |
|label_pbtxt|数据label| |
|pipeline_config|模型配置文件|如不提供，则使用model_name指定模型原始配置文件|
|images_path|数据图片存放位置|与xmls_path、train_txt、eval_txt一起提供，生成TFrecord以及pbtxt文件，用于模型训练|
|xmls_path|数据标签xml文件存放位置| |
|train_txt|训练文件名文件| |
|eval_txt|测试文件名文件| |


train.py 推荐使用命令行运行(使用IED运行时，因cuda版本问题，存在GPU调用失败情况)
