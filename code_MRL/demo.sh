DATASET='CIFAR10' # ('CIFAR10', 'CIFAR100')
MODEL='ResNet18' # ('ResNet18' 'MobileNetV2' 'SENet' 'DPN26')
ALPHA='0.1'
EPS='0.001' # (0.001, 0.0001)
NAME='0'

python main_baseline.py --model ${MODEL} --dataset ${DATASET} --name=${NAME}\

python main_MRL.py --pretrained-model ${MODEL} --model ${MODEL} --dataset ${DATASET} --name=${NAME}\
 --eps ${EPS} --alpha ${ALPHA}\