from resnet import get_pretrained_resnet
import logging
from enum import Enum
from torch.utils.data import DataLoader
import torch
from resnet import PalmNet
import orchestrate.io
import numpy as np
import math
from resnet_stanford_cars_cli import StanfordCarsCLI, Hyperparameters, CLI
from sigopt_python.sigopt import Connection


class SigoptExperimentCLI(StanfordCarsCLI):
    def __init__(self):
        super().__init__()

    def load_datasets(self, parsed_cli_arguments):
        return super().load_datasets(parsed_cli_arguments)

    def evaluate_model(self, sigopt_suggestion_arguments, assignments, training_data, validation_data, sigopt_conn, experiment_id, suggestion):
        logging.info("loading pretrained model and establishing model characteristics")

        resnet_pretrained_model = get_pretrained_resnet(sigopt_suggestion_arguments[CLI.FREEZE_WEIGHTS.value],
                                                        sigopt_suggestion_arguments[CLI.NUM_CLASSES.value],
                                                        sigopt_suggestion_arguments[CLI.MODEL.value])
        cross_entropy_loss = torch.nn.CrossEntropyLoss()

        sgd_optimizer = torch.optim.SGD(resnet_pretrained_model.parameters(),
                                        lr=np.exp(assignments[Hyperparameters.LEARNING_RATE.value]),
                                        momentum=assignments[Hyperparameters.MOMENTUM.value],
                                        weight_decay=np.exp(assignments[Hyperparameters.WEIGHT_DECAY.value]),
                                        nesterov=assignments[Hyperparameters.NESTEROV.value])
        learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(sgd_optimizer, mode='min',
                                                                             factor=assignments[Hyperparameters.LEARNING_RATE_SCHEDULER.value],
                                                                             patience=assignments[Hyperparameters.SCEDULER_RATE.value],
                                                                             verbose=True)

        logging.info("training model")
        palm_net = PalmNet(epochs=sigopt_suggestion_arguments[CLI.EPOCHS.value], gd_optimizer=sgd_optimizer, model=resnet_pretrained_model,
                           loss_function=cross_entropy_loss,
                           learning_rate_scheduler=learning_rate_scheduler,
                           validation_frequency=sigopt_suggestion_arguments[CLI.VALIDATION_FREQUENCY.value],
                           torch_checkpoint_location=sigopt_suggestion_arguments[CLI.CHECKPOINT.value],
                           model_checkpointing=sigopt_suggestion_arguments[CLI.CHECKPOINT_FREQUENCY.value])

        trained_model, validation_metric = palm_net.train_model(training_data=DataLoader(training_data,
                                                                                         batch_size=assignments[Hyperparameters.BATCH_SIZE.value],
                                                                                         shuffle=True),
                                                                validation_data=DataLoader(validation_data,
                                                                                           batch_size=assignments[Hyperparameters.BATCH_SIZE.value],
                                                                                           shuffle=True),
                                                                number_of_labels=sigopt_suggestion_arguments[
                                                                    CLI.NUM_CLASSES.value],
                                                                sigopt_conn=sigopt_conn,
                                                                experiment_id=experiment_id,
                                                                suggestion=suggestion)
        return trained_model, validation_metric

    def run(self, sigopt_suggestion_arguments, training_data, validation_data):

        conn = Connection(client_token=sigopt_suggestion_arguments[CLI.SIGOPT_TOKEN.value])

        # experiment = conn.experiments().create(
        #     name='Stanford Car Training Monitor',
        #     # Define which parameters you would like to tune
        #     parameters=[
        #         dict(
        #           name="batch_size",
        #           bounds=dict(
        #             min=4,
        #             max=8
        #             ),
        #           type="int"
        #           ),
        #         dict(
        #           name="brightness",
        #           bounds=dict(
        #             min=0,
        #             max=10
        #             ),
        #           type="double"
        #           ),
        #         dict(
        #           name="contrast",
        #           bounds=dict(
        #             min=0,
        #             max=100
        #             ),
        #           type="double"
        #           ),
        #         dict(
        #           name="hue",
        #           bounds=dict(
        #             min=-0.5,
        #             max=0.5
        #             ),
        #           type="double"
        #           ),
        #         dict(
        #           name="learning_rate",
        #           bounds=dict(
        #             min=-9.21,
        #             max=0
        #             ),
        #           type="double"
        #           ),
        #         dict(
        #           name="learning_rate_scheduler",
        #           bounds=dict(
        #             min=0,
        #             max=0.99
        #             ),
        #           type="double"
        #           ),
        #         dict(
        #           name="momentum",
        #           bounds=dict(
        #             min=0.001,
        #             max=0.9
        #             ),
        #           type="double"
        #           ),
        #         dict(
        #           name="nesterov",
        #           categorical_values=[
        #             dict(
        #               name="True",
        #               enum_index=1
        #               ),
        #             dict(
        #               name="False",
        #               enum_index=2
        #               )
        #             ],
        #           type="categorical"
        #           ),
        #         dict(
        #           name="saturation",
        #           bounds=dict(
        #             min=0,
        #             max=100
        #             ),
        #           type="double"
        #           ),
        #         dict(
        #           name="scheduler_rate",
        #           bounds=dict(
        #             min=0,
        #             max=20
        #             ),
        #           type="int"
        #           ),
        #         dict(
        #           name="weight_decay",
        #           bounds=dict(
        #             min=-11.3306,
        #             max=0
        #             ),
        #           type="double"
        #           )
        #         ],
        #     metrics=[dict(name='val_accuracy')],
        #     parallel_bandwidth=1,
        #     observation_budget=30,
        #     project="stanford-car-tm",
        #     training_monitor=dict(
        #         max_checkpoints=35,  # required, cannot exceed 200
        #         early_stopping_criteria=[
        #             dict(
        #               type='convergence',  # Only permitted value during alpha testing
        #               name='Look Back 4 Steps',
        #               metric='val_accuracy',
        #               lookback_checkpoints=2,
        #               min_checkpoints=3,  # Minimum checkpoints before stopping criteria is considered
        #             ),
        #           ],
        #     ),
        # )

        experiment = conn.experiments(sigopt_suggestion_arguments[CLI.EXPERIMENT_ID.value]).fetch()

        # Run the Optimization Loop until the Observation Budget is exhausted
        while experiment.progress.observation_count < experiment.observation_budget:
            print("Evaluating Model")
            suggestion = conn.experiments(experiment.id).suggestions().create()
            print(suggestion.assignments)
            _, value = self.evaluate_model(sigopt_suggestion_arguments, suggestion.assignments, training_data, validation_data, conn, experiment.id, suggestion)
            # conn.experiments(experiment.id).observations().create(
            #     suggestion=suggestion.id,
            #     value=value,
            # )

            # Update the experiment object
            experiment = conn.experiments(experiment.id).fetch()

        print("Experiment Completed")


if __name__ == "__main__":
    sigopt_experiment_cli = SigoptExperimentCLI()
    sigopt_experiment_cli.run_all()