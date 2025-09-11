python collect_data.py --start_index=0 --image_dir="image_train" --state_dir="state_train" --num=5000
python collect_data.py --start_index=0 --image_dir="image_test" --state_dir="state_test" --num=1000
python train.py