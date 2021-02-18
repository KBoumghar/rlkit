

from railrl.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from torch.utils.tensorboard import SummaryWriter
from railrl.core import logger


class PPOAlgorithm(TorchBatchRLAlgorithm):
    
    def __init__(
            self,            *args,
            **kwargs
    ):

        super().__init__(*args, **kwargs)
        # Create a tensorboard writer to log things.
        self.writer = SummaryWriter(logger.get_snapshot_dir() + '/tensorboard')
    
    def _train(self):
        
        out = super()._train()
        
        if ((self.epoch % 5) == 0):
            self.render_video_and_add_to_tensorboard("eval_video")
        
        return out
        
    def render_video_and_add_to_tensorboard(self, tag):
        import numpy as np
        import pdb
#         log.debug("{}".format("render_video_and_add_to_tensorboard"))
        
        path = self.eval_data_collector.collect_new_paths(
            self.max_path_length,
            self.num_eval_steps_per_epoch,
            discard_incomplete_paths=True,
        )
#         pdb.set_trace()
#         timer.stamp('evaluation sampling')
        
#         path = self.exploration_data_collectors.model_exploration_collector.collect_rendered_rollout(self.max_path_length)
#         video = np.stack(path['rendering'], axis=0)
#         pdb.set_trace()
        video = np.array([ x['rendering'] for x in  path[0]['env_infos']])
        video = np.moveaxis(video, -1, -3)[None]
        self.writer.add_video(tag, video, self.epoch, fps=8)