import unittest
from trainer import Trainer


class TestTrainer(unittest.TestCase):
    def test_training(self):
        trainer = Trainer()
        try:
            trainer.import_data('test2') # todo vytvorit data pre tento test
            # trainer.index = 'training_2021_01_28_13_07'
            trainer.train()
        finally:
            trainer.delete_index()


if __name__ == '__main__':
    unittest.main()
