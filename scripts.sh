python collect_data.py --start_index=0 --num=20000 --image_dir="image_train" --state_dir="state_train" --texture_dir="texture_train"
python collect_data.py --start_index=0 --num=1000 --image_dir="image_test" --state_dir="state_test" --texture_dir="texture_test"
python train.py
python test.py