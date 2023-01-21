train_path=/userhome/dataset/aim19_class/train
test_path=/userhome/dataset/aim19
dataset_name=aim
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
--patch_size 128 \
--tensorboard \
--max_epoch 800 \
--width_list 0.6 0.5 0.4 \
--operation train \
--name "mbcnn_aim19_stage1"
}

train_stage2(){
python main.py \
--traindata_path $train_path \
--testdata_path $test_path \
--dataset $dataset_namse \
--batchsize 4 \
--lr 1e-5 \
--patch_size 256 \
--tensorboard \
--max_epoch 800 \
--width_list 0.6 0.5 0.4 \
--resume './result/MBCNN_aim19/1pth_folder/ckpt_best.pth'\
--operation train \
--name "mbcnn_fhdmi_stage2" 
}


#test_supernet
#train
train_stage1
train_stage2
