python train_harddrop_needle.py --root_dir /home/youhanlee/project/RANZCR_kaggle/ --exp exp0_after_pre_needle --DEBUG False --use_amp True --batch_size 36 --valid_batch_size 36 --num_workers 4 --model_name efficientnet_b3 --CUDA_VISIBLE_DEVICES "4" --image_size 512 --fold_id 0 --n_epochs 50 --pretrained_model ./weights_pre/efficientnet_b3_fold0_pre_best_loss.pth