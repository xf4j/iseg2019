from __future__ import division
import os, re
import numpy as np
from skimage.transform import resize, warp, AffineTransform
from skimage import measure
from transforms3d.euler import euler2mat
from transforms3d.affines import compose
import tensorflow as tf
import SimpleITK as sitk
import h5py

from evaluations import compute_metric

class BaseModel(object):
    def save(self, step, model_name='main'):
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)
        
    def load(self, model_name='main'):
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            return True, counter
        else:
            print("Failed to find a checkpoint")
            return False, 0
        
    def estimate_mean_std(self):
        means = []
        stds = []
        for i in range(400):
            n = np.random.choice(len(self.training_subjects))
            images, _ = self.read_training_inputs(self.training_root, self.training_subjects[n])
            means.append(np.mean(images, axis=(0, 1, 2)))
            stds.append(np.std(images, axis=(0, 1, 2)))
        means = np.asarray(means)
        stds = np.asarray(stds)
        return np.mean(means, axis=0), np.mean(stds, axis=0)
        
    def read_training_inputs(self, root, sub, augmentation=False):
        T1 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(root, sub + '-T1.hdr'))).astype(np.float32)
        T2 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(root, sub + '-T2.hdr'))).astype(np.float32)
        label = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(root, sub + '-label.hdr')))
        nclass = self.nclass
        class_labels = self.class_labels
        
        if augmentation and self.use_mr_augmentation:
            for z in range(1, nclass):
                T1[label == class_labels[z]] = T1[label == class_labels[z]] * np.random.uniform(0.9, 1.1)
                T2[label == class_labels[z]] = T2[label == class_labels[z]] * np.random.uniform(0.9, 1.1)
            T1 = self.add_noise(T1)
            T2 = self.add_noise(T2)
                
        T1 = (T1 - np.amin(T1)) / (np.amax(T1) - np.amin(T1))
        T2 = (T2 - np.amin(T2)) / (np.amax(T2) - np.amin(T2))
        
        full_size = T1.shape
        im_size = self.im_size
        
        x_range = full_size[0] - im_size[0]
        y_range = full_size[1] - im_size[1]
        z_range = full_size[2] - im_size[2]
        
        '''
        # Get sampling prob
        x_offset = int(im_size[0] / 2)
        y_offset = int(im_size[1] / 2)
        z_offset = int(im_size[2] / 2)
        
        p = label[x_offset : x_offset + x_range, y_offset : y_offset + y_range, z_offset : z_offset + z_range]
        p = p.astype(np.float32)
        p[p > 0] = 0.8
        p[p == 0] = 0.2
        p = p.flatten() / np.sum(p)
        
        o = np.random.choice(x_range * y_range * z_range, p=p)
        '''
        o = np.random.choice(x_range * y_range * z_range)
        
        x_start, y_start, z_start = np.unravel_index(o, (x_range, y_range, z_range))
        
        T1_extracted = T1[x_start : x_start + im_size[0], y_start : y_start + im_size[1], z_start : z_start + im_size[2]]
        T2_extracted = T2[x_start : x_start + im_size[0], y_start : y_start + im_size[1], z_start : z_start + im_size[2]]
        label_extracted = label[x_start : x_start + im_size[0], y_start : y_start + im_size[1], z_start : z_start + im_size[2]]
        
        if np.random.uniform() > 0.5:
            # Flip left and right
            T1_extracted = T1_extracted[..., ::-1]
            T2_extracted = T2_extracted[..., ::-1]
            label_extracted = label_extracted[..., ::-1]
        
        if augmentation:
            # augmentation
            translation = [0, 0, 0] # No translation is necessary since the location is at random
            rotation = euler2mat(np.random.uniform(-5, 5) / 180.0 * np.pi, np.random.uniform(-5, 5) / 180.0 * np.pi,
                                 np.random.uniform(-5, 5) / 180.0 * np.pi, 'sxyz')
            scale = [np.random.uniform(0.8, 1.2), np.random.uniform(0.8, 1.2), np.random.uniform(0.8, 1.2)]
            warp_mat = compose(translation, rotation, scale)
            tform_coords = self.get_tform_coords(im_size)
            w = np.dot(warp_mat, tform_coords)
            w[0] = w[0] + im_size[0] / 2
            w[1] = w[1] + im_size[1] / 2
            w[2] = w[2] + im_size[2] / 2
            warp_coords = w[0:3].reshape(3, im_size[0], im_size[1], im_size[2])

            T1_extracted = warp(T1_extracted, warp_coords)
            T2_extracted = warp(T2_extracted, warp_coords)
        
            final_labels = np.empty(im_size + [nclass], dtype=np.float32)
            for z in range(1, nclass):
                temp = warp((label_extracted == class_labels[z]).astype(np.float32), warp_coords)
                temp[temp < 0.5] = 0
                temp[temp >= 0.5] = 1
                final_labels[..., z] = temp
            final_labels[..., 0] = np.amax(final_labels[..., 1:], axis=3) == 0   
        else:
            final_labels = np.zeros(im_size + [nclass], dtype=np.float32)
            for z in range(nclass):
                final_labels[label_extracted == class_labels[z], z] = 1
         
        final_images = np.stack((T1_extracted, T2_extracted), axis=-1)
        return final_images, final_labels
    
    def read_testing_inputs(self, root, sub, nstride=1):
        T1 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(root, sub + '-T1.hdr'))).astype(np.float32)
        T2 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(root, sub + '-T2.hdr'))).astype(np.float32)
        T1 = (T1 - np.amin(T1)) / (np.amax(T1) - np.amin(T1))
        T2 = (T2 - np.amin(T2)) / (np.amax(T2) - np.amin(T2))
        
        full_size = T1.shape
        im_size = self.im_size
        
        # pad first
        x_stride = int(im_size[0] / nstride)
        y_stride = int(im_size[1] / nstride)
        z_stride = int(im_size[2] / nstride)
        
        x_step = int(np.ceil(full_size[0] / x_stride)) + 1 - nstride
        y_step = int(np.ceil(full_size[1] / y_stride)) + 1 - nstride
        z_step = int(np.ceil(full_size[2] / z_stride)) + 1 - nstride
        
        x_pad = (x_step - 1) * x_stride + im_size[0]
        y_pad = (y_step - 1) * y_stride + im_size[1]
        z_pad = (z_step - 1) * z_stride + im_size[2]
        
        info = {
            'full_size': full_size,
            'pad_size': (x_pad, y_pad, z_pad),
            'step': (x_step, y_step, z_step),
            'stride': (x_stride, y_stride, z_stride)
        }
        
        pad = ((0, x_pad - full_size[0]), (0, y_pad - full_size[1]), (0, z_pad - full_size[2]))
        
        T1 = np.pad(T1, pad, mode='constant')
        T2 = np.pad(T2, pad, mode='constant')
        
        images = np.empty([x_step * y_step * z_step] + im_size + [self.input_features], dtype=np.float32)
        
        for ix in range(x_step):
            for iy in range(y_step):
                for iz in range(z_step):
                    o = ix * y_step * z_step + iy * z_step + iz
                    T1_extracted = T1[ix * x_stride : ix * x_stride + im_size[0], iy * y_stride : iy * y_stride + im_size[1],
                                      iz * z_stride : iz * z_stride + im_size[2]]
                    T2_extracted = T2[ix * x_stride : ix * x_stride + im_size[0], iy * y_stride : iy * y_stride + im_size[1],
                                      iz * z_stride : iz * z_stride + im_size[2]]
                    images[o] = np.stack((T1_extracted, T2_extracted), axis=-1)
                    
        return images, info
        
    def add_noise(self, input_im):
        m = np.mean(input_im)
        s = np.std(input_im)
        output_im = (input_im - m ) / s
        output_im = output_im + np.random.normal(scale=np.random.uniform(0, 0.2), size=im.shape)
        output_im = output_im * s + m
        return output_im
        
    def get_tform_coords(self, im_size):
        coords0, coords1, coords2 = np.mgrid[:im_size[0], :im_size[1], :im_size[2]]
        coords = np.array([coords0 - im_size[0] / 2, coords1 - im_size[1] / 2, coords2 - im_size[2] / 2])
        return np.append(coords.reshape(3, -1), np.ones((1, np.prod(im_size))), axis=0)
    
    def clean_contour(self, in_contour, is_prob=False):
        if is_prob:
            pred = (in_contour >= 0.5).astype(np.float32)
        else:
            pred = in_contour
        labels = measure.label(pred)
        area = []
        for l in range(1, np.amax(labels) + 1):
            area.append(np.sum(labels == l))
        out_contour = in_contour
        out_contour[np.logical_and(labels > 0, labels != np.argmax(area) + 1)] = 0
        return out_contour

    def restore_label(self, probs, info):
        full_size = info['full_size']
        pad_size = info['pad_size']
        x_step, y_step, z_step = info['step']
        x_stride, y_stride, z_stride = info['stride']
        im_size = probs.shape[1 : -1]
        
        label_prob = np.zeros(pad_size + (probs.shape[-1],), dtype=np.float32)
        label_count = np.zeros(pad_size, dtype=np.float32)
        for ix in range(x_step):
            for iy in range(y_step):
                for iz in range(z_step):
                    o = ix * y_step * z_step + iy * z_step + iz
                    label_prob[ix * x_stride : ix * x_stride + im_size[0], iy * y_stride : iy * y_stride + im_size[1],
                               iz * z_stride : iz * z_stride + im_size[2], :] += probs[o]
                    label_count[ix * x_stride : ix * x_stride + im_size[0], iy * y_stride : iy * y_stride + im_size[1],
                                iz * z_stride : iz * z_stride + im_size[2]] += 1
        
        label_prob = label_prob / np.tile(np.expand_dims(label_count, axis=3), (1, 1, 1, label_prob.shape[-1]))
        label_prob = label_prob[:full_size[0], :full_size[1], :full_size[2], :]
        label = np.argmax(label_prob, axis=3)
        return label
    
    def train(self):
        self.sess.run(tf.global_variables_initializer())
        
        train_writer = tf.summary.FileWriter(os.path.join(self.log_dir, self.model_dir, 'train'), self.sess.graph)
        if self.testing_during_training:
            test_writer = tf.summary.FileWriter(os.path.join(self.log_dir, self.model_dir, 'test'))
        
        merged = tf.summary.merge([self.loss_summary, self.accuracy_summary, self.dice_summary])
        
        counter = 0
        
        gradients = tf.gradients(self.loss, self.images)[0]
        
        for epo in range(self.epoch):
            training_subjects = np.random.permutation(self.training_subjects)
            if epo == 0:
                mean, std = self.estimate_mean_std()
                self.sess.run([self.mean.assign(mean), self.std.assign(std)])
                
            for f in range(len(training_subjects) // self.batch_size):
                images = np.empty((self.batch_size, self.im_size[0], self.im_size[1], self.im_size[2], self.input_features),
                                  dtype=np.float32)
                labels = np.empty((self.batch_size, self.im_size[0], self.im_size[1], self.im_size[2], self.nclass),
                                  dtype=np.float32)
                for b in range(self.batch_size):
                    order = f * self.batch_size + b
                    images[b], labels[b] = self.read_training_inputs(self.training_root, training_subjects[order])
                    
                images = (images - mean) / std
                ## adversarial training
                if self.use_adversarial_training:
                    EPS = 2 / 255.0 / std
                    x = images + np.random.uniform(-EPS, EPS, images.shape)
                    
                    
                    for i in range(5):
                        grad = self.sess.run(gradients, feed_dict={ self.images: images,
                                                                    self.labels: labels,
                                                                    self.is_training: True,
                                                                    self.keep_prob: 1 })
                        x = np.add(x, 0.1 * np.sign(grad), out=x, casting='unsafe')
                        x = np.clip(x, images - EPS, images + EPS)
                        
                _, train_loss, summary = self.sess.run([self.optimizer, self.loss, merged],
                                                       feed_dict={ self.images: x,
                                                                   self.labels: labels,
                                                                   self.is_training: True,
                                                                   self.keep_prob: self.dropout })
                train_writer.add_summary(summary, counter)
                counter += 1
                if np.mod(counter, 1000) == 0:
                    self.save(counter)
                
            # Test during training, for simplicity, this relies on the label information to extract images so that it
            # only tells whether the trained model can segment the roi well given the extracted images
            if self.testing_during_training and (np.mod(epo, 100) == 0 or epo == self.epoch - 1):
                test_dice_mean = 0
                for testing_subject in self.testing_subjects:
                    output_label = self.run_test(testing_subject)
                    
                    img = sitk.ReadImage(os.path.join(self.testing_root, testing_subject + '-label.hdr'))
                    gt_label = sitk.GetArrayFromImage(img).astype(np.float32)
                    for roi in range(self.nclass - 1):
                        gt = (gt_label == self.class_labels[roi + 1]).astype(np.float32)
                        model = (output_label == roi + 1).astype(np.float32)
                        test_dice_mean += np.sum(gt * model) / (np.sum(gt) + np.sum(model)) * 2.0
                        
                test_dice_mean = test_dice_mean / len(self.testing_subjects) / (self.nclass - 1)
                test_dice_summary = tf.Summary(value=[tf.Summary.Value(tag='dice', simple_value=test_dice_mean)])
                test_writer.add_summary(test_dice_summary, counter)
                    
        # Save in the end
        self.save(counter)
    
    def run_test(self, testing_subject, flip_augmentation=False):
        nstride = self.test_nstride
        mean, std = self.sess.run([self.mean, self.std])
        test_batch = 1
        images = np.empty((test_batch, self.im_size[0], self.im_size[1], self.im_size[2], self.input_features), 
                          dtype=np.float32)
        all_images, info = self.read_testing_inputs(self.testing_root, testing_subject, nstride)
        
        patch_size = all_images.shape[0]
        
        pad = int(np.ceil(all_images.shape[0] / test_batch)) * test_batch
        if pad > all_images.shape[0]:
            all_images = np.pad(all_images, ((0, pad - all_images.shape[0]), (0, 0), (0, 0), (0, 0)), mode='constant')
            
        all_images = (all_images - mean) / std
            
        all_probs = np.empty((all_images.shape[:-1] + (self.nclass,)), dtype=np.float32)
            
        for n in range(all_images.shape[0] // test_batch):
            for b in range(test_batch):
                images[b] = all_images[n * test_batch + b]
            probs = self.sess.run(self.probs, feed_dict = { self.images: images, self.is_training: True, self.keep_prob: 1 })
            if flip_augmentation:
                images = images[..., ::-1, :]
                probs_flip = self.sess.run(self.probs, feed_dict = { self.images: images, self.is_training: True,
                                                                     self.keep_prob: 1 })
                probs = (probs + probs_flip[..., ::-1, :]) / 2.0
            '''
            if self.test_voting:
                pred = np.argmax(probs, axis=4)
                probs.fill(0)
                for c in range(probs.shape[-1]):
                    probs[..., pred == c, c] = 1
            '''
            all_probs[n * test_batch : (n + 1) * test_batch] = probs
                
        output_label = self.restore_label(all_probs[:patch_size, ...], info)
        return output_label
            
    def test(self, output_path, test_with_gt):
        if not self.load()[0]:
            raise Exception("No model is found, please train first")
            
        if self.use_mr_augmentation:
            output_path = os.path.join(output_path, 'mraug')
        else:
            output_path = os.path.join(output_path, 'noaug')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        nstride = self.test_nstride
        
        if test_with_gt:
            mean_dice = []
            mean_msd = []
            mean_hd95 = []
            for roi in range(self.nclass - 1):
                mean_dice.append(0)
                mean_msd.append(0)
                mean_hd95.append(0)
        
        for testing_subject in self.testing_subjects:
            output_label = self.run_test(testing_subject, flip_augmentation=True)
            
            label_writing = np.empty(output_label.shape, dtype=np.uint8)
            for roi in range(self.nclass):
                label_writing[output_label == roi] = self.class_labels[roi]
                
            T1_img = sitk.ReadImage(os.path.join(self.testing_root, testing_subject + '-T1.hdr'))
            label_img = sitk.GetImageFromArray(label_writing)
            label_img.CopyInformation(T1_img)
            sitk.WriteImage(label_img, os.path.join(output_path, testing_subject + '-label.hdr'))
            
            if test_with_gt:
                img = sitk.ReadImage(os.path.join(self.testing_root, testing_subject + '-label.hdr'))
                gt_label = sitk.GetArrayFromImage(img).astype(np.float32)
                for roi in range(self.nclass - 1):
                    gt = (gt_label == self.class_labels[roi + 1]).astype(np.float32)
                    model = (output_label == roi + 1).astype(np.float32)
                    dice, msd, hd95 = compute_metric(gt, model, self.voxel_spacing)
                    print(str(dice) + ' ' +  str(msd) + ' ' + str(hd95))
                    mean_dice[roi] = mean_dice[roi] + dice
                    mean_msd[roi] = mean_msd[roi] + msd
                    mean_hd95[roi] = mean_hd95[roi] + hd95
                
        if test_with_gt:
            for roi in range(self.nclass - 1):
                mean_dice[roi] = mean_dice[roi] / len(self.testing_subjects)
                mean_msd[roi] = mean_msd[roi] / len(self.testing_subjects)
                mean_hd95[roi] = mean_hd95[roi] / len(self.testing_subjects)
            print(mean_dice)
            print(mean_msd)
            print(mean_hd95)