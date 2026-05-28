import torch
import numpy as np
import os
import pickle
import argparse
from copy import deepcopy
from itertools import repeat
from tqdm import tqdm
import wandb
from utils import load_data
from utils import compute_dict_mean, set_seed, detach_dict
from policy import ChunkSeqPolicy, CNNMLPPolicy, DiffusionPolicy

import IPython
e = IPython.embed


def main(args):
    set_seed(1)
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_steps = args['num_steps']
    eval_every = args['eval_every']
    validate_every = args['validate_every']
    save_every = args['save_every']
    resume_ckpt_path = args['resume_ckpt_path']

    if is_eval:
        config_path = os.path.join(ckpt_dir, 'config.pkl')
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        run_checkpoint_validation(config)
        return

    if not args.get('dataset_dir'):
        raise ValueError('Training requires --dataset_dir')
    if not args.get('episode_len'):
        raise ValueError('Training requires --episode_len (positive int)')
    if not args.get('camera_names'):
        raise ValueError('Training requires --camera_names (comma-separated)')

    dataset_dir = args['dataset_dir']
    episode_len = args['episode_len']
    camera_names = [c.strip() for c in args['camera_names'].split(',') if c.strip()]
    if not camera_names:
        raise ValueError('--camera_names must list at least one camera key (comma-separated).')
    stats_dir = None
    sample_weights = None
    train_ratio = args['train_ratio']
    name_filter = lambda n: True

    # fixed parameters (match SeqVAE / dataset layout)
    state_dim = 16
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'CHUNK_SEQ':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         'vq': args['use_vq'],
                         'vq_class': args['vq_class'],
                         'vq_dim': args['vq_dim'],
                         'action_dim': 18,
                         'no_encoder': args['no_encoder'],
                         }
    elif policy_class == 'Diffusion':

        policy_config = {'lr': args['lr'],
                         'camera_names': camera_names,
                         'action_dim': 16,
                         'observation_horizon': 1,
                         'action_horizon': 8,
                         'prediction_horizon': args['chunk_size'],
                         'num_queries': args['chunk_size'],
                         'num_inference_timesteps': 10,
                         'ema_power': 0.75,
                         'vq': False,
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError

    config = {
        'num_steps': num_steps,
        'eval_every': eval_every,
        'validate_every': validate_every,
        'save_every': save_every,
        'ckpt_dir': ckpt_dir,
        'resume_ckpt_path': resume_ckpt_path,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'load_pretrain': args['load_pretrain'],
        'dataset_dir': dataset_dir,
        'batch_size': batch_size_train,
        'skip_mirrored_data': args['skip_mirrored_data'],
        'train_ratio': train_ratio,
    }

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    config_path = os.path.join(ckpt_dir, 'config.pkl')
    expr_name = os.path.basename(os.path.normpath(ckpt_dir))
    wandb.init(project=os.environ.get('WANDB_PROJECT', 'mimoco'), reinit=True, name=expr_name)
    wandb.config.update(config)
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)

    train_dataloader, val_dataloader, stats = load_data(dataset_dir, name_filter, camera_names, batch_size_train, batch_size_val, args['chunk_size'], args['skip_mirrored_data'], config['load_pretrain'], policy_class, stats_dir_l=stats_dir, sample_weights=sample_weights, train_ratio=train_ratio)

    # save dataset stats
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_step, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ step{best_step}')
    wandb.finish()


def make_policy(policy_class, policy_config):
    if policy_class == 'CHUNK_SEQ':
        policy = ChunkSeqPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    elif policy_class == 'Diffusion':
        policy = DiffusionPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'CHUNK_SEQ':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'Diffusion':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def run_checkpoint_validation(config, ckpt_name='policy_last.ckpt', max_batches=80):
    """Mean validation loss from saved config and weights (no environment rollout)."""
    if 'dataset_dir' not in config:
        raise ValueError(
            'config.pkl is missing dataset_dir; re-train with this codebase to write a complete config.'
        )
    set_seed(config['seed'])
    dataset_dir = config['dataset_dir']
    camera_names = config['camera_names']
    batch_size = config['batch_size']
    policy_class = config['policy_class']
    chunk_size = config['policy_config']['num_queries']
    name_filter = lambda n: True
    stats_dir = None
    sample_weights = None
    train_ratio = config.get('train_ratio', 0.99)
    skip_mirrored = config.get('skip_mirrored_data', False)
    load_pretrain = config.get('load_pretrain', False)

    _, val_dataloader, _ = load_data(
        dataset_dir, name_filter, camera_names, batch_size, batch_size, chunk_size,
        skip_mirrored, load_pretrain, policy_class,
        stats_dir_l=stats_dir, sample_weights=sample_weights, train_ratio=train_ratio,
    )

    ckpt_path = os.path.join(config['ckpt_dir'], ckpt_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = make_policy(policy_class, config['policy_config'])
    policy.deserialize(torch.load(ckpt_path, map_location=device))
    policy.to(device)
    policy.eval()

    losses = []
    with torch.inference_mode():
        for batch_idx, data in enumerate(val_dataloader):
            fd = forward_pass(data, policy, device)
            losses.append(float(fd['loss'].item()))
            if batch_idx + 1 >= max_batches:
                break
    print(f'Loaded {ckpt_path}, mean val loss over {len(losses)} batches: {np.mean(losses):.6f}')


def forward_pass(data, policy, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_data, qpos_data, action_data, is_pad = data
    image_data = image_data.to(device)
    qpos_data = qpos_data.to(device)
    action_data = action_data.to(device)
    is_pad = is_pad.to(device)
    return policy(qpos_data, image_data, action_data, is_pad)


def train_bc(train_dataloader, val_dataloader, config):
    num_steps = config['num_steps']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    eval_every = config['eval_every']
    validate_every = config['validate_every']
    save_every = config['save_every']

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    if config['load_pretrain']:
        pretrain_ckpt = os.environ.get('MIMOCO_PRETRAIN_CKPT', '')
        if not pretrain_ckpt or not os.path.isfile(pretrain_ckpt):
            raise FileNotFoundError(
                'load_pretrain is set but MIMOCO_PRETRAIN_CKPT is missing or not a file. '
                'Export your own pretrain checkpoint path into this env var.'
            )
        loading_status = policy.deserialize(torch.load(pretrain_ckpt))
        print(f'loaded! {loading_status}')
    if config['resume_ckpt_path'] is not None:
        loading_status = policy.deserialize(torch.load(config['resume_ckpt_path']))
        print(f'Resume policy from: {config["resume_ckpt_path"]}, Status: {loading_status}')
    policy.cuda()
    policy_device = next(policy.parameters()).device
    optimizer = make_optimizer(policy_class, policy)

    min_val_loss = np.inf
    best_ckpt_info = None
    
    train_dataloader = repeater(train_dataloader)
    for step in tqdm(range(num_steps+1)):
        # validation
        if step % validate_every == 0:
            print('validating')

            with torch.inference_mode():
                policy.eval()
                validation_dicts = []
                for batch_idx, data in enumerate(val_dataloader):
                    forward_dict = forward_pass(data, policy, policy_device)
                    validation_dicts.append(forward_dict)
                    if batch_idx > 50:
                        break

                validation_summary = compute_dict_mean(validation_dicts)

                epoch_val_loss = validation_summary['loss']
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (step, min_val_loss, deepcopy(policy.serialize()))
            for k in list(validation_summary.keys()):
                validation_summary[f'val_{k}'] = validation_summary.pop(k)            
            wandb.log(validation_summary, step=step)
            print(f'Val loss:   {epoch_val_loss:.5f}')
            summary_string = ''
            for k, v in validation_summary.items():
                summary_string += f'{k}: {v.item():.3f} '
            print(summary_string)
                
        # evaluation
        if (step > 0) and (step % eval_every == 0):
            # first save then eval
            ckpt_name = f'policy_step_{step}_seed_{seed}.ckpt'
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            torch.save(policy.serialize(), ckpt_path)

        # training
        policy.train()
        optimizer.zero_grad()
        data = next(train_dataloader)
        forward_dict = forward_pass(data, policy, policy_device)
        # backward
        loss = forward_dict['loss']
        loss.backward()
        optimizer.step()
        wandb.log(forward_dict, step=step) # not great, make training 1-2% slower

        if step % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_step_{step}_seed_{seed}.ckpt')
            torch.save(policy.serialize(), ckpt_path)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.serialize(), ckpt_path)

    best_step, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_step_{best_step}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at step {best_step}')

    return best_ckpt_info

def repeater(data_loader):
    epoch = 0
    for loader in repeat(data_loader):
        for data in loader:
            yield data
        print(f'Epoch {epoch} done')
        epoch += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--eval',
        action='store_true',
        help='Load ckpt_dir/config.pkl and report mean validation loss',
    )
    parser.add_argument('--ckpt_dir', action='store', type=str, help='Checkpoint directory', required=True)
    parser.add_argument('--policy_class', type=str, default='CHUNK_SEQ')
    parser.add_argument('--task_name', type=str, default='mimoco_run', help='Run label for logging')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default=None,
        help='HDF5 dataset directory (required for training)',
    )
    parser.add_argument(
        '--episode_len',
        type=int,
        default=0,
        help='Max steps per episode (required for training)',
    )
    parser.add_argument(
        '--camera_names',
        type=str,
        default='',
        help='Comma-separated camera keys matching HDF5',
    )
    parser.add_argument('--train_ratio', type=float, default=0.99)
    parser.add_argument('--load_pretrain', action='store_true', default=False)
    parser.add_argument('--eval_every', action='store', type=int, default=500, help='eval_every', required=False)
    parser.add_argument('--validate_every', action='store', type=int, default=500, help='validate_every', required=False)
    parser.add_argument('--save_every', action='store', type=int, default=500, help='save_every', required=False)
    parser.add_argument('--resume_ckpt_path', action='store', type=str, help='resume_ckpt_path', required=False)
    parser.add_argument('--skip_mirrored_data', action='store_true')
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--use_vq', action='store_true')
    parser.add_argument('--vq_class', action='store', type=int, help='vq_class')
    parser.add_argument('--vq_dim', action='store', type=int, help='vq_dim')
    parser.add_argument('--no_encoder', action='store_true')
    
    main(vars(parser.parse_args()))
