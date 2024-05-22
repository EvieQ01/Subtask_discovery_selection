# Create color dataset
python make_synthetic.py --lag 10
python make_synthetic.py --lag 3

# Create driving dataset
python make_driving.py # You can change the number of trajectories, or the length of each trajectory in the script.


# Most importantly, conduct the CI test for the following two structures:
# v-structure 1 (st and at)
# v-structure 2 (gt and at+1)
python test_ci.py --data color_seq --lag 3
python test_ci.py --data color_seq --lag 10
python test_ci.py --data driving_seq      