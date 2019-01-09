from rlkit.samplers.util import rollout
from rlkit.torch.core import PyTorchModule
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import joblib
import uuid
from rlkit.core import logger
from rlkit.torch.dqn.dqn import DQN
import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger

filename = str(uuid.uuid4())
variant = dict(
    algo_params=dict(
        num_epochs=500,
        num_steps_per_epoch=1000,
        num_steps_per_eval=1000,
        batch_size=128,
        max_path_length=200,
        discount=0.90,
        epsilon=0.2,
        tau=0.002,
        learning_rate=0.001,
        hard_update_period=1000,
        save_environment=True,
    ),
)


def resume_training(args):
    data = joblib.load(args.file)
    #policy = data['policy']
    env = data['env']
    epoch = data['epoch']
    qf = data['qf']
    target_qf = data['target_qf']
    opt_state = data['optimizer_state']
    algorithm = DQN(
        env,
        training_env=env,
        qf=qf,
        target_qf = target_qf,
        **variant['algo_params']
    )
    algorithm.to(ptu.device)
    algorithm.train(start_epoch=epoch)
    print("Policy loaded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')

    args = parser.parse_args()
    log_name = args.file.split('/')[-3]
    print("logname", log_name)
    setup_logger(log_name, variant=variant)

    resume_training(args)
