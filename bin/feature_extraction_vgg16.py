import tensorflow as tf
from tensorflow import flags
from datetime import datetime

FLAGS = flags.FLAGS

if __name__ == '__main__':
    flags.DEFINE_string('root_model_dir', './models/vgg16',
                        'Root directory for saving tensorflow models')
    flags.DEFINE_string('root_log_dir', './tf_logs/vgg16',
                        'Root directory for saving tensorboard logs')

class VGG16(object):
    """Construct the VGG 16 model"""
    def __init__(self):
        pass

    def generate_model_log_dirs(self, root_model_dir, root_log_dir):
        datetime_now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        log_dir = '{}/log-{}/'.format(FLAGS.root_log_dir, datetime_now)
        model_dir = '{}/model-{}/'.format(FLAGS.root_model_dir, datetime_now)
        return log_dir, model_dir

    def run(self):
        log_dir, model_dir = self.generate_model_log_dirs(
            FLAGS.root_model_dir, FLAGS.root_log_dir)
        print(log_dir, model_dir)

def main(argv=None):
    vgg_model = VGG16()
    vgg_model.run()

if __name__ == '__main__':
    tf.app.run()