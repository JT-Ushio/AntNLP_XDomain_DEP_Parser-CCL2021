exper_name=$1
conda_env=$2
gpu_id=$3
default_cfg=$4
argu_list=${@:5}
res=`screen -ls`
if [[ ${res} =~ "${exper_name}" ]]
then
    echo "screen $exper_name already created."
else
    screen -dmS $exper_name
    echo "create screen $exper_name"
fi

# setup conda_env & execute_dir
screen -r $exper_name -X stuff "conda activate $conda_env \n"
screen -r $exper_name -X stuff 'xdomain \n'

# mkdir ckpt_dir
ckpt_dir=../ckpts/$exper_name
screen -r $exper_name -X stuff  " if [ ! -d $ckpt_dir ]; then \n \
mkdir $ckpt_dir \n \
fi \n"

# copy default config
exper_cfg="${ckpt_dir}/run.cfg"
screen -r $exper_name -X stuff "cp $default_cfg $exper_cfg \n"

# run training scripts
echo "CUDA_VISIBLE_DEVICES=$gpu_id python train.py $argu_list \n"

screen -r $exper_name -X stuff "CUDA_VISIBLE_DEVICES=$gpu_id python train.py $argu_list \n"

echo 'cmd finish!'