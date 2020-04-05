import os, sys
import numpy as np
import pickle
import tensorflow as tf
import configparser

from models.UNet3D import UNet3D
from models.DenseUNet3D import DenseUNet3D

flags = tf.app.flags
flags.DEFINE_integer("phase", 1, "0 - train cv, 1 - test cv, 2 - train final, 3 - deploy final")
flags.DEFINE_string("model", "UNet3D", "Model name [UNet3D]")
flags.DEFINE_string("config", "models/UNet3D.ini", "Model config file [models/UNet3D.ini]")
flags.DEFINE_string("train_dir", "../Training", "Training dir")
flags.DEFINE_string("test_dir", "../Validation", "Testing dir")
FLAGS = flags.FLAGS

def main(_):
    phase = FLAGS.phase
    model_name = FLAGS.model
    config_file = FLAGS.config
    train_dir = FLAGS.train_dir
    test_dir = FLAGS.test_dir
    
    if model_name != 'UNet3D' and model_name != 'DenseUNet3D':
        raise Exception("Unsupported model")
    
    config = configparser.ConfigParser()
    config.read(config_file)
    
    if phase == 0 or phase == 1:
        testing_during_training = True
        training_root = train_dir
        testing_root = train_dir
        if os.path.exists(os.path.join(train_dir, 'files.log')):
            with open(os.path.join(train_dir, 'files.log'), 'r') as f:
                training_subjects, testing_subjects = pickle.load(f)
        else:
            all_subjects = [name.replace('-label.hdr', '') for name in os.listdir(train_dir) if name.endswith('-label.hdr')]
            all_subjects = [s for s in all_subjects if not s.startswith('.')]
            np.random.shuffle(all_subjects)
            n_training = int(len(all_subjects) * 3 / 5)
            training_subjects = all_subjects[:n_training]
            testing_subjects = all_subjects[n_training:]
            with open(os.path.join(train_dir, 'files.log'), 'w') as f:
                pickle.dump([training_subjects, testing_subjects], f)
    else:
        testing_during_training = False
        training_root = train_dir
        testing_root = test_dir
        training_subjects = [name.replace('-label.hdr', '') for name in os.listdir(train_dir) if name.endswith('-label.hdr')]
        training_subjects = [s for s in training_subjects if not s.startswith('.')]
        testing_subjects = [name.replace('-T1.hdr', '') for name in os.listdir(test_dir) if name.endswith('-T1.hdr')]
        testing_subjects = [s for s in testing_subjects if not s.startswith('.')]
        
    if phase == 0 or phase == 2:
        is_train = True
    else:
        is_train = False
        
    checkpoint_dir = './checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    log_dir = './logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # First step
    run_config = tf.ConfigProto()
    with tf.Session(config=run_config) as sess:
        # The multi-label dice loss is unstable
        if model_name == 'UNet3D':
            Model = UNet3D
        elif model_name == 'DenseUNet3D':
            Model = DenseUNet3D
            
        model = Model(sess, checkpoint_dir=checkpoint_dir, log_dir=log_dir, training_root=training_root,
                      testing_root=testing_root, training_subjects=training_subjects, testing_subjects=testing_subjects,
                      testing_during_training=testing_during_training, model_config=config['Model'])
        
        if is_train:
            model.train()
        else:
            if phase == 1:
                output_dir = './output/cv'
                test_with_gt = True
            else:
                output_dir = './output/test'
                test_with_gt = False
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                    
            model.test(output_dir, test_with_gt)

    tf.reset_default_graph()
    
if __name__ == '__main__':
    tf.app.run()