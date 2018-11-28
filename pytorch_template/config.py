class Config(object):
    pass


class TrainConfig(object):
    trainer = Config()
    trainer.batch_size = 100
    trainer.learning_rate = 0.001
    trainer.evaluation_interval = 1000
    trainer.checkpoint_interval = 1000

    def __init__(self, debug=False):
        if debug:
            self.trainer.evaluation_interval = 5