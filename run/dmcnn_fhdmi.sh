train_path=/userhome/dataset/fhdmi_class/train
test_path=/userhome/dataset/fhdmi
dataset_name=fhdmi
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
--patch_size 384 \
--tensorboard \
--max_epoch 200 \
--width_list 0.75 0.5 0.25 \
--operation train \
--name "dmcnn_fhdmi"
}

train
