#python -m torch.distributed.launch --nproc_per_node=3 train_dist.py --root_dir /home/ubuntu/lyh/RANZCR_kaggle/ --exp exp0 --DEBUG False --use_amp True --batch_size 28 --valid_batch_size 28 --num_workers 4 --model_name tf_efficientnet_b6  --CUDA_VISIBLE_DEVICES "0,1,2" --image_size 512 --fold_id 0 --model_dir ./weights_1 --log_dir ./logs_1 #--debug False

python -m torch.distributed.launch --nproc_per_node=3 train_dist.py --root_dir /home/ubuntu/lyh/RANZCR_kaggle/ --exp exp0 --DEBUG False --use_amp True --batch_size 28 --valid_batch_size 28 --num_workers 4 --model_name tf_efficientnet_b6  --CUDA_VISIBLE_DEVICES "0,1,2" --image_size 512 --fold_id 1 --model_dir ./weights_1 --log_dir ./logs_1 #--debug False

kill $(ps aux | grep "train_dist.py" | grep -v grep | awk '{print $2}')
kill $(ps aux | grep "train_dist.py" | grep -v grep | awk '{print $2}')

python -m torch.distributed.launch --nproc_per_node=3 train_dist.py --root_dir /home/ubuntu/lyh/RANZCR_kaggle/ --exp exp0 --DEBUG False --use_amp True --batch_size 28 --valid_batch_size 28 --num_workers 4 --model_name tf_efficientnet_b6  --CUDA_VISIBLE_DEVICES "0,1,2" --image_size 512 --fold_id 2 --model_dir ./weights_1 --log_dir ./logs_1 #--debug False

kill $(ps aux | grep "train_dist.py" | grep -v grep | awk '{print $2}')
kill $(ps aux | grep "train_dist.py" | grep -v grep | awk '{print $2}')

python -m torch.distributed.launch --nproc_per_node=3 train_dist.py --root_dir /home/ubuntu/lyh/RANZCR_kaggle/ --exp exp0 --DEBUG False --use_amp True --batch_size 28 --valid_batch_size 28 --num_workers 4 --model_name tf_efficientnet_b6  --CUDA_VISIBLE_DEVICES "0,1,2" --image_size 512 --fold_id 3 --model_dir ./weights_1 --log_dir ./logs_1 #--debug False

kill $(ps aux | grep "train_dist.py" | grep -v grep | awk '{print $2}')
kill $(ps aux | grep "train_dist.py" | grep -v grep | awk '{print $2}')

python -m torch.distributed.launch --nproc_per_node=3 train_dist.py --root_dir /home/ubuntu/lyh/RANZCR_kaggle/ --exp exp0 --DEBUG False --use_amp True --batch_size 28 --valid_batch_size 28 --num_workers 4 --model_name tf_efficientnet_b6  --CUDA_VISIBLE_DEVICES "0,1,2" --image_size 512 --fold_id 4 --model_dir ./weights_1 --log_dir ./logs_1 #--debug False

kill $(ps aux | grep "train_dist.py" | grep -v grep | awk '{print $2}')
kill $(ps aux | grep "train_dist.py" | grep -v grep | awk '{print $2}')
