from sigopt_python.sigopt import Connection

conn = Connection(client_token="EYIWCEHVMWDGQVVWRZXMTJHAOQREPJVVPDEFYUMAFOITHBUF")

# suggestion = conn.experiments(87108).suggestions().create()
# training_run = conn.experiments(87108).training_runs().create(suggestion=suggestion.id)


experiment = conn.experiments().create(
    name='Stanford Car Training Monitor',
    # Define which parameters you would like to tune
    parameters=[
        dict(
          name="batch_size",
          bounds=dict(
            min=4,
            max=8
            ),
          type="int"
          ),
        dict(
          name="brightness",
          bounds=dict(
            min=0,
            max=10
            ),
          type="double"
          ),
        dict(
          name="contrast",
          bounds=dict(
            min=0,
            max=100
            ),
          type="double"
          ),
        dict(
          name="hue",
          bounds=dict(
            min=-0.5,
            max=0.5
            ),
          type="double"
          ),
        dict(
          name="learning_rate",
          bounds=dict(
            min=-9.21,
            max=0
            ),
          type="double"
          ),
        dict(
          name="learning_rate_scheduler",
          bounds=dict(
            min=0,
            max=0.99
            ),
          type="double"
          ),
        dict(
          name="momentum",
          bounds=dict(
            min=0.001,
            max=0.9
            ),
          type="double"
          ),
        dict(
          name="nesterov",
          categorical_values=[
            dict(
              name="True",
              enum_index=1
              ),
            dict(
              name="False",
              enum_index=2
              )
            ],
          type="categorical"
          ),
        dict(
          name="saturation",
          bounds=dict(
            min=0,
            max=100
            ),
          type="double"
          ),
        dict(
          name="scheduler_rate",
          bounds=dict(
            min=0,
            max=20
            ),
          type="int"
          ),
        dict(
          name="weight_decay",
          bounds=dict(
            min=-11.3306,
            max=0
            ),
          type="double"
          )
        ],
    metrics=[dict(name='val_accuracy')],
    parallel_bandwidth=1,
    observation_budget=30,
    project="stanford-car-tm",
    training_monitor=dict(
        max_checkpoints=35,  # required, cannot exceed 200
        early_stopping_criteria=[
          dict(
            type='convergence',  # Only permitted value during alpha testing
            name='Look Back 4 Steps',
            metric='val_accuracy',
            lookback_checkpoints=4,
            min_checkpoints=5,  # Minimum checkpoints before stopping criteria is considered
          ),
        ],
    ),
)