test_path=/userhome/dataset/fhdmi
dataset_name=fhdmi
arch=DMCNN
export CUDA_VISIBLE_DEVICES="1"

test(){
python main.py \
--arch MBCNN \
--testdata_path $test_path \
--dataset $dataset_name \
--Test_pretrained_path './ckpt/mbcnn_fhdmi.pth' \
--batchsize 1 \
--tensorboard \
--width_list 0.75 0.5 0.25 \
--operation test \
--name "test"
}

test
