test_path=/userhome/dataset/aim19
dataset_name=aim
arch=DMCNN
export CUDA_VISIBLE_DEVICES="0"

test(){
python main.py \
--arch MBCNN \
--testdata_path $test_path \
--dataset $dataset_name \
--Test_pretrained_path './ckpt/mbcnn_aim.pth' \
--batchsize 1 \
--tensorboard \
--width_list 0.6 0.5 0.4 \
--operation test \
--name "test"
}

test
