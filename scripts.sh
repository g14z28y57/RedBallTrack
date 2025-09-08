python collect_data.py --start_index=0 --num=10000 --image_dir="image_train" --state_dir="state_train"
python collect_data.py --start_index=0 --num=500 --image_dir="image_test" --state_dir="state_test"
python train.py