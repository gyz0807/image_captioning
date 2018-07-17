import tensorflow as tf
from tensorflow import flags
from datetime import datetime

FLAGS = flags.FLAGS

if __name__ == '__main__':
    flags.DEFINE_string('root_model_dir', './models/feature_extraction',
                        'Root directory for saving tensorflow models')
    flags.DEFINE_string('root_log_dir', './tf_logs/feature_extraction',
                        'Root directory for saving tensorboard logs')

class VGG16_SPP(object):
    """Construct a VGG 16 model with Spatial Pyramic Pooling"""
    def __init__(self):
        pass

    def generate_model_log_dirs(self, root_model_dir, root_log_dir):
        datetime_now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        log_dir = '{}/log-{}/'.format(FLAGS.root_log_dir, datetime_now)
        model_dir = '{}/model-{}/'.format(FLAGS.root_model_dir, datetime_now)
        return log_dir, model_dir

    def double_conv_layers(self, input_tensor, filters, kernel_size, strides, padding):
        conv_layer_1 = tf.layers.conv2d(input_tensor, filters,
            kernel_size, strides, padding)
        conv_layer_2 = tf.layers.conv2d(conv_layer_1, filters,
            kernel_size, strides, padding)
        return conv_layer_2
    
    def generate_graph(self, log_dir):
        tf.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default():
            with tf.variable_scope('input'):
                image_input = tf.placeholder(tf.float32, [None, 224, 224, 3], name='image_input')
            with tf.variable_scope('vgg16'):
                with tf.variable_scope('block1'):
                    conv_block_1 = self.double_conv_layers(
                        image_input, 64, [3, 3], [1, 1], 'SAME')
                    conv_block_1_pooled = tf.layers.max_pooling2d(
                        conv_block_1, [2, 2], [2, 2], 'SAME')
                with tf.variable_scope('block2'):
                    conv_block_2 = self.double_conv_layers(
                        conv_block_1_pooled, 128, [3, 3], [1, 1], 'SAME')
                    conv_block_2_pooled = tf.layers.max_pooling2d(
                        conv_block_2, [2, 2], [2, 2], 'SAME')
                with tf.variable_scope('block3'):
                    conv_block_3 = self.double_conv_layers(
                        conv_block_2_pooled, 256, [3, 3], [1, 1], 'SAME')
                    conv_block_3_pooled = tf.layers.max_pooling2d(
                        conv_block_3, [2, 2], [2, 2], 'SAME')
                with tf.variable_scope('block4'):
                    conv_block_4 = self.double_conv_layers(
                        conv_block_3_pooled, 512, [3, 3], [1, 1], 'SAME')
                    conv_block_4_pooled = tf.layers.max_pooling2d(
                        conv_block_4, [2, 2], [2, 2], 'SAME')
                with tf.variable_scope('block5'):
                    conv_block_5 = self.double_conv_layers(
                        conv_block_4_pooled, 512, [3, 3], [1, 1], 'SAME')
            with tf.variable_scope('spatial_pyramid_pooling'):
                pass

        file_writer = tf.summary.FileWriter(log_dir, graph)

    def run(self):
        log_dir, model_dir = self.generate_model_log_dirs(
            FLAGS.root_model_dir, FLAGS.root_log_dir)
        self.generate_graph(log_dir)

def main(argv=None):
    vgg_model = VGG16_SPP()
    vgg_model.run()

if __name__ == '__main__':
    tf.app.run()
