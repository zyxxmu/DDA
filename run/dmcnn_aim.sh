train_path=/userhome/dataset/aim19_class/train
test_path=/userhome/dataset/aim19
dataset_name=aim
arch=DMCNN
export CUDA_VISIBLE_DEVICES="0"

train(){
python main.py \
--arch DMCNN \
--traindata_path $train_path \
--testdata_path $test_path \
--dataset $dataset_name \
--batchsize 4 \
--lr 1e-4 \
--patch_size 256 \
--tensorboard \
--max_epoch 200 \
--width_list 0.6 0.5 0.4 \
--operation train \
--name "dmcnn_aim19"
}

train
