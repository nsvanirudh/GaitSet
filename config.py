conf = {
    "WORK_PATH": "./work",
    "CUDA_VISIBLE_DEVICES": "0",
    "data": {
        'dataset_path': "/scratch0/snanduri/silh_T_3",
        'resolution': '64',
        'dataset': 'CASIA-B',
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        # For more detail, please refer to
        # function: utils.data_loader.load_data
        'pid_num': 73,
        'pid_shuffle': False,
        'normalize': 'area'
    },
    "model": {
        'hidden_dim': 256, #256
        'lr': 5e-5,
        'hard_or_full_trip': 'full',
        'batch_size': (8, 16),
        'restore_iter': 160000,
        'total_iter': 240000,
        'margin': 0.2,
        'num_workers': 3,
        'frame_num': 30,
        'model_name': 'GaitSet_SilhPotion_1period_onlyArea',
    },
}
