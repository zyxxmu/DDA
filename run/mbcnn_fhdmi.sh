train_path=/userhome/dataset/fhdmi_class/train
test_path=/userhome/dataset/fhdmi
dataset_name=fhdmi
arch=MBCNN
export CUDA_VISIBLE_DEVICES="0"

train_stage1(){
python main.py \
--arch MBCNN \
--traindata_path $train_path \
--testdata_path $test_path \
--dataset $dataset_name \
--batchsize 16 \
--lr 1e-4 \
--patch_size 192 \
--tensorboard \
--max_epoch 800 \
--width_list 0.75 0.5 0.25 \
--operation train \
--name "mbcnn_fhdmi_stage1"
}

train_stage2(){
python main.py \
--traindata_path $train_path \
--testdata_path $test_path \
--dataset $dataset_name \
--batchsize 4 \
--lr 1e-5 \
--patch_size 384 \
--tensorboard \
--max_epoch 800 \
--width_list 0.75 0.5 0.25 \
--resume './result/MBCNN_fhdmi/1pth_folder/ckpt_best.pth'\
--operation train \
--name "mbcnn_fhdmi_stage2" 
}


#test_supernet
#train
train_stage1
train_stage2
