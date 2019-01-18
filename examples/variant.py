VARIANT = dict(
    algo_params=dict(
        num_epochs=500,
        num_steps_per_epoch=1000,
        num_steps_per_eval=1000,
        num_updates_per_epoch=1000,
        batch_size=128,
        discount=0.90,
        epsilon=0.2,
        tau=0.002,
        learning_rate=0.001,
        hard_update_period=1000,
        save_environment=True,  
        #collection_mode='batch',
        max_path_length=30
    ),
)