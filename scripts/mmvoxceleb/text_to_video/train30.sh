python3 train.py --name tys_30_0306 \
    --image_text_folder /home/sunmeng/mmvid/data/mmvoxceleb/tys_train \
    --dataset video_text --batch_size 4 \
    --text_seq_len 50 --fixed_language_model roberta-large \
    --iters 300000 \
    --num_workers 0 \
    --log_every 500 --sample_every 5000 \
    --n_sample 4 --n_per_sample 4 --num_visuals 0 \
    --num_targets 8 --frame_num 8 --frame_step 4 \
    --image_size 128 --beta_msm 7 \
    --dropout_vc 0.4 --dist_url tcp://localhost:10001 \
    --vae_path ./pretrained_models/vae_vox.ckpt --rel_no_fully_masked \
    --mask_predict_steps 10 20 30 --mask_predict_steps1 20
