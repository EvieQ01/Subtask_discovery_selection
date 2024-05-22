
# Ours
python run_hierIL_baselines.py --env_name  "KitchenMetaEnv-v0" --n_traj 1000 --algo option_gail --device cuda --selected_train_id '[6, 14]' --selected_test_id '[5]' --option_gt 0 --option_nmf 1 --seed 0 --render 0
# Option-GAIL
python run_hierIL_baselines.py --env_name  "KitchenMetaEnv-v0" --n_traj 1000 --algo option_gail --device cuda --selected_train_id '[6, 14]' --selected_test_id '[5]' --option_gt 0 --option_nmf 0 --seed 0 --render 0
# DI-GAIL
python run_hierIL_baselines.py --env_name  "KitchenMetaEnv-v0" --n_traj 1000 --algo DI_gail --device cuda --selected_train_id '[6, 14]' --selected_test_id '[5]' --option_gt 0 --option_nmf 0 --seed 0 --render 0
# H-AIRL
python run_hierIL_baselines.py --env_name  "KitchenMetaEnv-v0" --n_traj 1000 --algo hier_airl --device cuda --selected_train_id '[6, 14]' --selected_test_id '[5]' --option_gt 0 --option_nmf 0 --seed 0 --render 0


# Ours
python run_hierIL_baselines.py --env_name  "KitchenMetaEnv-v0" --n_traj 1000 --algo option_gail --device cuda --selected_train_id '[6, 14]' --selected_test_id '[13]' --option_gt 0 --option_nmf 1 --seed 0 --render 0
# Option-GAIL
python run_hierIL_baselines.py --env_name  "KitchenMetaEnv-v0" --n_traj 1000 --algo option_gail --device cuda --selected_train_id '[6, 14]' --selected_test_id '[13]' --option_gt 0 --option_nmf 0 --seed 0 --render 0
# DI-GAIL
python run_hierIL_baselines.py --env_name  "KitchenMetaEnv-v0" --n_traj 1000 --algo DI_gail --device cuda --selected_train_id '[6, 14]' --selected_test_id '[13]' --option_gt 0 --option_nmf 0 --seed 0 --render 0
# H-AIRL
python run_hierIL_baselines.py --env_name  "KitchenMetaEnv-v0" --n_traj 1000 --algo hier_airl --device cuda --selected_train_id '[6, 14]' --selected_test_id '[13]' --option_gt 0 --option_nmf 0 --seed 0 --render 0
