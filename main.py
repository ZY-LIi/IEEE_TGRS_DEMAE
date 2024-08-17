from loop_train_test import loop_train_test


def resolve_hp(hp: dict):
    return hp.get('run_times'), hp.get('num_PC'), hp.get('train_num'), \
           hp.get('patch_size'), hp.get('batch_size'), hp.get('lr'), \
           hp.get('epoch'), hp.get('pretrained_weights_path'), \
           hp.get('pretrained_timesteps'), hp.get('finetuned_timesteps'), \
           hp.get('finetuned_mask_ratio')


def PU_experiment(hp: dict):
    run_times, num_PC, train_num, patch_size, batch_size, lr, epoch, \
    pretrained_weights_path, pretrained_timesteps, finetuned_timesteps, finetuned_mask_ratio = resolve_hp(hp)
    loop_train_test('PU', run_times, num_PC, train_num, patch_size, batch_size, lr, epoch,
                    pretrained_weights_path, pretrained_timesteps, finetuned_timesteps, finetuned_mask_ratio)


def Salinas_experiment(hp: dict):
    run_times, num_PC, train_num, patch_size, batch_size, lr, epoch, \
    pretrained_weights_path, pretrained_timesteps, finetuned_timesteps, finetuned_mask_ratio = resolve_hp(hp)
    loop_train_test('Salinas', run_times, num_PC, train_num, patch_size, batch_size, lr, epoch,
                    pretrained_weights_path, pretrained_timesteps, finetuned_timesteps, finetuned_mask_ratio)


def HU_experiment(hp: dict):
    run_times, num_PC, train_num, patch_size, batch_size, lr, epoch, \
    pretrained_weights_path, pretrained_timesteps, finetuned_timesteps, finetuned_mask_ratio = resolve_hp(hp)
    loop_train_test('Houston', run_times, num_PC, train_num, patch_size, batch_size, lr, epoch,
                    pretrained_weights_path, pretrained_timesteps, finetuned_timesteps, finetuned_mask_ratio)


def LongKou_experiment(hp: dict):
    run_times, num_PC, train_num, patch_size, batch_size, lr, epoch, \
    pretrained_weights_path, pretrained_timesteps, finetuned_timesteps, finetuned_mask_ratio = resolve_hp(hp)
    loop_train_test('LongKou', run_times, num_PC, train_num, patch_size, batch_size, lr, epoch,
                    pretrained_weights_path, pretrained_timesteps, finetuned_timesteps, finetuned_mask_ratio)


if __name__ == '__main__':
    hyperparameter_pu = {
        'run_times': 1,
        'num_PC': 36,
        'train_num': 5,
        'patch_size': 11,
        'batch_size': 45,
        'lr': 1e-3,
        'epoch': 160,
        'pretrained_weights_path': './save/pretrained_weights/PU_pretrained_weights_patch11_pc36_timesteps200_mask75.pt',
        'pretrained_timesteps': 200,
        'finetuned_timesteps': 150,
        'finetuned_mask_ratio': 0.75,
    }
    PU_experiment(hp=hyperparameter_pu)

    # hyperparameter_salinas = {
    #     'run_times': 1,
    #     'num_PC': 40,
    #     'train_num': 5,
    #     'patch_size': 11,
    #     'batch_size': 80,
    #     'lr': 1e-3,
    #     'epoch': 100,
    #     'pretrained_weights_path': './save/pretrained_weights/Salinas_pretrained_weights_patch11_pc40_timesteps300_mask75.pt',
    #     'pretrained_timesteps': 300,
    #     'finetuned_timesteps': 150,
    #     'finetuned_mask_ratio': 0.75,
    # }
    # Salinas_experiment(hp=hyperparameter_salinas)

    # hyperparameter_hu = {
    #     'run_times': 1,
    #     'num_PC': 36,
    #     'train_num': 5,
    #     'patch_size': 11,
    #     'batch_size': 75,
    #     'lr': 3e-3,
    #     'epoch': 150,
    #     'pretrained_weights_path': './save/pretrained_weights/Houston_pretrained_weights_patch11_pc36_timesteps200_mask75.pt',
    #     'pretrained_timesteps': 200,
    #     'finetuned_timesteps': 50,
    #     'finetuned_mask_ratio': 0.75,
    # }
    # HU_experiment(hp=hyperparameter_hu)

    # hyperparameter_longkou = {
    #     'run_times': 1,
    #     'num_PC': 36,
    #     'train_num': 5,
    #     'patch_size': 11,
    #     'batch_size': 45,
    #     'lr': 1e-3,
    #     'epoch': 150,
    #     'pretrained_weights_path': './save/pretrained_weights/LongKou_pretrained_weights_patch11_pc36_timesteps300_mask75.pt',
    #     'pretrained_timesteps': 300,
    #     'finetuned_timesteps': 100,
    #     'finetuned_mask_ratio': 0.75,
    # }
    # LongKou_experiment(hp=hyperparameter_longkou)
