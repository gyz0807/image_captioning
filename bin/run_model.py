import tensorflow as tf
from tensorflow import flags
from datetime import datetime

FLAGS = flags.FLAGS

if __name__ == '__main__':
    flags.DEFINE_string('root_model_dir', './models/image_caption',
                        'Root directory for saving tensorflow models')
    flags.DEFINE_string('root_log_dir', './tf_logs/image_caption',
                        'Root directory for saving tensorboard logs')

class ImageCaptioning(object):
    def __init__(self):
        pass
    
    def generate_model_log_dirs(self, root_model_dir, root_log_dir):
        datetime_now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        log_dir = '{}/log-{}/'.format(FLAGS.root_log_dir, datetime_now)
        model_dir = '{}/model-{}/'.format(FLAGS.root_model_dir, datetime_now)
        return log_dir, model_dir

    def generate_graph(self):
        tf.reset_default_graph()
        graph = tf.Graph()

    def run(self):
        log_dir, model_dir = self.generate_model_log_dirs(
            FLAGS.root_model_dir, FLAGS.root_log_dir)

def main(argv=None):
    image_captioning_model = ImageCaptioning()
    image_captioning_model.run()

if __name__ == '__main__':
    tf.app.run()