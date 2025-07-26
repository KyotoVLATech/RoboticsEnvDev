import logging
from typing import Any, Callable, Dict, Optional, Union
from tianshou.data import Collector
from tianshou.policy import BasePolicy
from tianshou.trainer.base import BaseTrainer
from tianshou.utils import LazyLogger

class DSRLTrainer(BaseTrainer):
    """Custom trainer that supports both on-policy and off-policy algorithms with video recording.
    
    This trainer extends BaseTrainer and automatically handles the differences between
    on-policy and off-policy training based on the algorithm specified in config.
    It also provides automatic video recording at specified intervals.
    
    :param str algorithm: The RL algorithm to use ('sac', 'ppo', 'dqn', 'a2c', etc.)
    :param int video_record_interval: Record training video every N epochs. Set to 0 to disable.
    :param Callable record_video_fn: Function to record training video with signature
        f(env, policy, config, epoch) -> None
    :param Dict config: Configuration dictionary containing training parameters
    :param other parameters: Same as BaseTrainer
    """
    
    def __init__(
        self,
        algorithm: str,
        video_record_interval: int = 10,
        record_video_fn: Optional[Callable] = None,
        config: Optional[Dict] = None,
        train_envs = None,
        **kwargs
    ):
        self.algorithm = algorithm.lower()
        self.video_record_interval = video_record_interval
        self.record_video_fn = record_video_fn
        self.config = config or {}
        self.train_envs = train_envs
        self.last_video_epoch = 0
        # Validate algorithm
        off_policy_algorithms = ['sac', 'dqn', 'ddpg', 'td3']
        on_policy_algorithms = ['ppo', 'a2c', 'trpo', 'vision_ppo']
        if self.algorithm not in off_policy_algorithms + on_policy_algorithms:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}.\nSupported: {off_policy_algorithms + on_policy_algorithms}")
        self.is_off_policy = self.algorithm in off_policy_algorithms
        
        # Determine learning_type based on algorithm
        learning_type = "offpolicy" if self.is_off_policy else "onpolicy"
        
        super().__init__(learning_type=learning_type, **kwargs)
        logging.info(f"DSRLTrainer initialized with algorithm: {self.algorithm}")
        logging.info(f"Training mode: {'off-policy' if self.is_off_policy else 'on-policy'}")
        if self.video_record_interval > 0:
            logging.info(f"Video recording enabled every {self.video_record_interval} epochs")

    def policy_update_fn(self, data: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Policy update function that handles both on-policy and off-policy algorithms."""
        assert self.train_collector is not None
        if self.is_off_policy:
            # Off-policy update (SAC, DQN, DDPG, TD3)
            for _ in range(round(self.update_per_step * result["n/st"])):
                self.gradient_step += 1
                losses = self.policy.update(self.batch_size, self.train_collector.buffer)
                self.log_update_data(data, losses)
        else:
            # On-policy update (PPO, A2C, TRPO)
            losses = self.policy.update(
                0,
                self.train_collector.buffer,
                batch_size=self.batch_size,
                repeat=self.repeat_per_collect,
            )
            self.train_collector.reset_buffer(keep_statistics=True)
            step = max([1] + [len(v) for v in losses.values() if isinstance(v, list)])
            self.gradient_step += step
            self.log_update_data(data, losses)

    def __next__(self):
        """Override __next__ to add video recording functionality."""
        result = super().__next__()
        if result is None:
            return result
        epoch, epoch_stat, info = result
        # Check if we should record video
        if (self.video_record_interval > 0 and self.record_video_fn is not None and epoch % self.video_record_interval == 0 and epoch > self.last_video_epoch and self.train_envs is not None):
            logging.info(f"Recording training video for epoch {epoch}")
            self.record_video_fn(self.train_envs, self.policy, self.config, epoch)
            self.last_video_epoch = epoch
        return result

def create_dsrl_trainer(
    algorithm: str,
    policy: BasePolicy,
    train_collector: Collector,
    test_collector: Optional[Collector],
    max_epoch: int,
    step_per_epoch: int,
    batch_size: int,
    config: Dict,
    record_video_fn: Optional[Callable] = None,
    train_envs = None,
    **kwargs
) -> DSRLTrainer:
    """Factory function to create DSRLTrainer with appropriate parameters for different algorithms.
    :param str algorithm: Algorithm name ('sac', 'ppo', 'dqn', etc.)
    :param BasePolicy policy: The policy to train
    :param Collector train_collector: Training data collector
    :param Collector test_collector: Test data collector (can be None)
    :param int max_epoch: Maximum number of training epochs
    :param int step_per_epoch: Steps per epoch
    :param int batch_size: Batch size for training
    :param Dict config: Configuration dictionary
    :param Callable record_video_fn: Video recording function
    :param train_envs: Training environments for video recording
    :param kwargs: Additional arguments passed to DSRLTrainer
    :return: Configured DSRLTrainer instance
    """
    # Set algorithm-specific defaults
    off_policy_algorithms = ['sac', 'dqn', 'ddpg', 'td3']
    on_policy_algorithms = ['ppo', 'a2c', 'trpo', 'vision_ppo']
    if algorithm.lower() in off_policy_algorithms:
        # Off-policy specific parameters
        trainer_kwargs = {
            'step_per_collect': config.get('step_per_collect', 10),
            'update_per_step': config.get('update_per_step', 1.0),
            'episode_per_test': config.get('episode_per_test', 10),
        }
    else:
        # On-policy specific parameters
        trainer_kwargs = {
            'step_per_collect': config.get('step_per_collect', 2048),
            'repeat_per_collect': config.get('repeat_per_collect', 10),
            'episode_per_test': config.get('episode_per_test', 10),
        }
    # Common parameters
    common_kwargs = {
        'train_fn': config.get('train_fn'),
        'test_fn': config.get('test_fn'),
        'stop_fn': config.get('stop_fn'),
        'save_best_fn': config.get('save_best_fn'),
        'save_checkpoint_fn': config.get('save_checkpoint_fn'),
        'resume_from_log': config.get('resume_from_log', False),
        'reward_metric': config.get('reward_metric'),
        'logger': config.get('logger', LazyLogger()),
        'verbose': config.get('verbose', True),
        'show_progress': config.get('show_progress', True),
        'test_in_train': config.get('test_in_train', True),
    }
    # Merge all kwargs
    trainer_kwargs.update(common_kwargs)
    trainer_kwargs.update(kwargs)
    return DSRLTrainer(
        algorithm=algorithm,
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=max_epoch,
        step_per_epoch=step_per_epoch,
        batch_size=batch_size,
        video_record_interval=config.get('video_record_interval', 10),
        record_video_fn=record_video_fn,
        config=config,
        train_envs=train_envs,
        **trainer_kwargs
    )


def dsrl_trainer(
    algorithm: str,
    policy: BasePolicy,
    train_collector: Collector,
    test_collector: Optional[Collector],
    max_epoch: int,
    step_per_epoch: int,
    batch_size: int,
    config: Dict,
    record_video_fn: Optional[Callable] = None,
    train_envs = None,
    **kwargs
) -> Dict[str, Union[float, str]]:
    """Wrapper function for DSRLTrainer.run() method.
    This function provides the same interface as tianshou's offpolicy_trainer and onpolicy_trainer
    but automatically handles both types based on the algorithm specified.
    :return: Training results dictionary
    """
    trainer = create_dsrl_trainer(
        algorithm=algorithm,
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=max_epoch,
        step_per_epoch=step_per_epoch,
        batch_size=batch_size,
        config=config,
        record_video_fn=record_video_fn,
        train_envs=train_envs,
        **kwargs
    )
    return trainer.run()
