## Motivation & Goal
The motivation of this small project is to demonstrate that a small properly-structured neural networks can learn how to track a target object.

## Establish the Virtual 3D Environment
A fixed-sized grey floor is placed in the level, and a red ball with fixed radius is placed randomly onto the floor. 
The camera position and focal point are also randomly placed each time when a photo shot is taken.

## Model Training
The input to the model are the photo shot, camera position and camera front. 
The model is trained to output the unit vector and distance from the camera position to ball position.

## Code Instruction
```
# Set up python environment
python -m venv venv
source venv/bin/activate

# Install packages
bash install.sh

# Generate 100 train data
python collect_data.py --start_index=0 --num=100 --image_dir="image_train" --state_dir="state_train"

# Generate test data
python collect_data.py --start_index=0 --num=100 --image_dir="image_test" --state_dir="state_test"

# Train model
python train.py

# Test Model
python test.py
```
