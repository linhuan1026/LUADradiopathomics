python run_infer.py \
--gpu='0' \
--nr_types=6 \
--type_info_path=type_info_pan.json \
--batch_size=16 \
--model_mode=fast \
--model_path=./weights/hovernet_fast_pannuke_type_tf2pytorch.tar \
--nr_inference_workers=8 \
--nr_post_proc_workers=16 \
tile \
--input_dir= /home/hero/disk/DATASET/test \
--output_dir=/home/hero/disk/DATASET/test_out \
--mem_usage=0.1 \
--draw_dot \
--save_qupath
