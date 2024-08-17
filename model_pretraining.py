# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import time
from loss_function import *
from data import linear_beta_schedule, HSI_LazyProcessing
from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)

def data_augmentation(patch):
    '''

    :param p: the probability of execute data augmentation
    :return:
    '''

    def vertical_rotation(patch):

        return np.flip(patch, axis=0)

    def horizontal_rotation(patch):

        return np.flip(patch, axis=1)

    def transpose(patch):

        return patch.transpose((1, 0, 2))

    patch = vertical_rotation(patch) if np.random.choice([0, 1], p=[1/2, 1/2]) else patch
    patch = horizontal_rotation(patch) if np.random.choice([0, 1], p=[1/2, 1/2]) else patch
    patch = transpose(patch) if np.random.choice([0, 1], p=[1/2, 1/2]) else patch

    return patch


def generate_batch(X, batch_size=64, patch_size=7, shuffle=True):
    '''

    :param X:
    :param batch_size:
    :param shuffle:
    :return:
    '''
    row, col, band = X.shape
    X = X.astype(np.float32)
    sample = np.ones((row - patch_size + 1, col - patch_size + 1))
    num_samples = int(sample.sum())
    patch_radius = patch_size // 2
    sample = np.pad(sample, ((patch_radius, patch_radius), (patch_radius, patch_radius)), "constant")
    sample = sample.reshape((row * col, -1))
    sequence = np.where(sample != 0)[0]

    if shuffle:
        random_state = np.random.RandomState()
        random_state.shuffle(sequence)

    for i in range(0, num_samples, batch_size):
        # batch_i represents the i-th element in current batch
        batch_i = sequence[np.arange(i, min(num_samples, i + batch_size))]
        batch_i_row = np.floor(batch_i * 1.0 / col).astype(np.int32)
        batch_i_col = (batch_i - batch_i_row * col).astype(np.int32)
        upper_edge, bottom_edge = (batch_i_row - patch_radius), (batch_i_row + patch_radius + 1)
        left_edge, right_edge = (batch_i_col - patch_radius), (batch_i_col + patch_radius + 1)

        patches = []
        for j in range(batch_i.size):
            patch = X[upper_edge[j]: bottom_edge[j], left_edge[j]: right_edge[j], :]
            patch = data_augmentation(patch)
            patches.append(patch)
        patches = np.array(patches)
        patches = np.transpose(patches, (0, 3, 1, 2))
        yield patches


def pretrain(model, X, dataset_name, patch_size=9, batch_size=64, epoches=10, lr=1e-2, mask_ratio=0.75, timesteps=200):

    train_loss_list = []

    # ---------- optimizer and loss implementation ---------- #

    loss_func = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # ---------- optimizer and loss implementation ---------- #

    model.train()
    model = model.to(device)
    # -------- diffusion betas -------- #
    betas = linear_beta_schedule(timesteps=timesteps) if timesteps > 0 else 0.
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0) if timesteps > 0 else torch.Tensor([1.])
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    # -------- diffusion betas -------- #

    t = time.time()
    min_loss = 1e6
    for epoch in range(epoches):
        train_loader = generate_batch(X, patch_size=patch_size, batch_size=batch_size)
        epoch_loss1 = 0
        epoch_loss2 = 0
        for step, x in enumerate(train_loader):
            x = torch.Tensor(x)
            B, C, H, W = x.shape

            num_tokens = H * W
            elements = np.concatenate(([1] * int(num_tokens * mask_ratio), [0] * (num_tokens - int(num_tokens * mask_ratio) - 1)), axis=0)
            mask = []
            for _ in range(B):
                elements = np.random.permutation(elements)
                mask.append(np.insert(elements, num_tokens // 2, 0))
            mask = torch.Tensor(np.array(mask)).bool()

            x = x.reshape(B, C, -1).transpose(1, 2)
            timestep = torch.randint(0, timesteps, (B,)).long() if timesteps > 0 else torch.zeros(B).long()
            noise = torch.randn_like(x)
            x_noise = sqrt_alphas_cumprod.gather(-1, timestep).reshape(B, *((1,) * 2)) * x + \
                      sqrt_one_minus_alphas_cumprod.gather(-1, timestep).reshape(B, *((1,) * 2)) * noise

            x_vis = x[~mask].reshape(B, -1, C)
            x_mask = x[mask].reshape(B, -1, C)
            x, x_noise, timestep = \
                x.to(device), x_noise.to(device), timestep.to(device)
            x_vis, x_mask = x_vis.to(device), x_mask.to(device)

            denoise, reconstruct = model(x_noise.float(), timestep, mask.to(device))
            optimizer.zero_grad()
            loss_dfs = loss_func(denoise, x_vis)
            loss_mae = loss_func(reconstruct, x_mask)
            epoch_loss1 += loss_dfs.item()
            epoch_loss2 += loss_mae.item()
            loss = loss_dfs + loss_mae
            loss.backward(retain_graph=True)
            optimizer.step()

        lr_scheduler.step()
        print("epoch: {}\tlr: {}\tloss_diffusion: {}\tloss_mae: {}".format(
            epoch, lr_scheduler.get_last_lr()[0], epoch_loss1 / (step + 1), epoch_loss2 / (step + 1)
        ))
        train_loss_list.append([epoch_loss1 / (step + 1), epoch_loss2 / (step + 1)])
        if np.sum(train_loss_list[-1]) < min_loss:
            min_loss = np.sum(train_loss_list[-1])
            torch.save(model.state_dict(),
                       './save/pretrained_weights/' + dataset_name + '_pretrained_weights_patch{}_pc{}_timesteps{}_mask{}.pt'.format(
                           int(patch_size), int(X.shape[-1]), int(timesteps), int(mask_ratio * 100)
                       ))
            print("Save model weights from epoch {}".format(epoch))

    train_loss_list = np.array(train_loss_list)
    plt.plot(train_loss_list[:, 0])
    plt.savefig(dataset_name + "_loss1_patch{}_pc{}_timesteps{}_mask{}.png".format(
        int(patch_size), int(X.shape[-1]), int(timesteps), int(mask_ratio * 100)
    ))
    plt.show()
    plt.plot(train_loss_list[:, 1])
    plt.savefig(dataset_name + "_loss2_patch{}_pc{}_timesteps{}_mask{}.png".format(
        int(patch_size), int(X.shape[-1]), int(timesteps), int(mask_ratio * 100)
    ))
    plt.show()

    print('Training model consumes %.2f seconds' % (time.time() - t))


if __name__ == "__main__":
    patch_size, pc, mask_ratio, timesteps = 11, 36, 0.75, 200
    X, _, _ = HSI_LazyProcessing(dataset_name='PU', n_pc=pc, no_processing=False, whiten=True)
    model = DEMAE_pretrain(dim=pc, patch_size=patch_size)
    pretrain(model, X, dataset_name='PU', patch_size=patch_size, batch_size=512, epoches=200, lr=1e-3,
             mask_ratio=mask_ratio, timesteps=timesteps)

    patch_size, pc, mask_ratio, timesteps = 11, 40, 0.75, 300
    X, _, _ = HSI_LazyProcessing(dataset_name='Salinas', n_pc=pc, no_processing=False, whiten=True)
    model = DEMAE_pretrain(dim=pc, patch_size=patch_size)
    pretrain(model, X, dataset_name='Salinas', patch_size=patch_size, batch_size=512, epoches=300, lr=1e-3,
             mask_ratio=mask_ratio, timesteps=timesteps)

    patch_size, pc, mask_ratio, timesteps = 11, 36, 0.75, 200
    X, _, _ = HSI_LazyProcessing(dataset_name='Houston', n_pc=pc, no_processing=False, whiten=True)
    model = DEMAE_pretrain(dim=pc, patch_size=patch_size)
    pretrain(model, X, dataset_name='Houston', patch_size=patch_size, batch_size=512, epoches=300, lr=1e-3,
             mask_ratio=mask_ratio, timesteps=timesteps)

    patch_size, pc, mask_ratio, timesteps = 11, 36, 0.75, 300
    X, _, _ = HSI_LazyProcessing(dataset_name='LongKou', n_pc=pc, no_processing=False, whiten=True)
    model = DEMAE_pretrain(dim=pc, patch_size=patch_size)
    pretrain(model, X, dataset_name='LongKou', patch_size=patch_size, batch_size=512, epoches=200, lr=1e-3,
             mask_ratio=mask_ratio, timesteps=timesteps)
