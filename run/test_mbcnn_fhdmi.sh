test_path=/userhome/dataset/fhdmi
dataset_name=fhdmi
arch=MBCNN
export CUDA_VISIBLE_DEVICES="1"

test(){
python main.py \
--arch MBCNN \
--testdata_path $test_path \
--dataset $dataset_name \
--Test_pretrained_path './ckpt/mbcnn_fhdmi.pth' \
--batchsize 1 \
--tensorboard \
--width_list 0.8 0.6 0.4 \
--operation test \
--name "test"
}

test
