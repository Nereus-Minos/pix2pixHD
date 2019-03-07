from activations import ins_norm, relu, pool, tanh
from conv_base import conv, conv_trans
from blocks import G_base, D_base, feat_loss, Save_im, load_data, res_block

import tensorflow as tf

import numpy as np

import os


class pix2pixHD:
    def __init__(self, opt):
        self.log_dir = opt.log_dir
        self.epoch = opt.epoch
        self.batch = opt.batch
        self.n_class = opt.n_class
        self.d_weight = 1/opt.num_d
        self.feat_weight = opt.feat_weight
        self.old_lr = opt.old_lr
        self.save_iter = opt.save_iter
        self.decay_ep = opt.decay_ep
        self.decay_weight = opt.decay_weight
        self.im_width = opt.im_width
        self.im_high = opt.im_high
        self.sace_ckpt_iter = opt.sace_ckpt_iter
        self.n_im = len(os.listdir(opt.data_dir))
        
        self.tf_record_dir = opt.tf_record_dir
        self.save_path = opt.save_path
        self.save_im_dir = opt.save_im_dir  
        self.ckpt_dir = opt.ckpt_dir
        self.label_dir = opt.label_dir
        self.inst_dir = opt.inst_dir
        
        self.label = tf.placeholder(tf.int32, [None, self.im_high, self.im_width])
        self.bound = tf.placeholder(tf.float32, [None, self.im_high, self.im_width])
        self.real_im = tf.placeholder(tf.float32, [None, self.im_high, self.im_width,3])
        self.k = tf.placeholder(tf.float32, [1])
        self.b = tf.placeholder(tf.float32, [None, self.im_high, self.im_width, 3])
        # process
        self.onehot = tf.one_hot(self.label, self.n_class)
        self.bound_ = tf.expand_dims(self.bound, 3)
        self.real_im_ = self.real_im/255
        
    # ############################  data_loader ################################# #
    def read_and_decode(self, filename):
        filename_queue = tf.train.string_input_producer([filename])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)   
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'Label': tf.FixedLenFeature([], tf.string),
                                               'Real': tf.FixedLenFeature([], tf.string),
                                               'Bound': tf.FixedLenFeature([], tf.string),
                                           })
        image_label = tf.decode_raw(features['Label'], tf.uint8)
        image_label = tf.reshape(image_label, [1024, 2048])

        image_real = tf.decode_raw(features['Real'], tf.uint8)
        image_real = tf.reshape(image_real, [1024, 2048, 3])

        image_bound = tf.decode_raw(features['Bound'], tf.uint8)
        image_bound = tf.reshape(image_bound, [1024, 2048])
        return image_label, image_real, image_bound
     # ############################################################################# #
    
    def build_G(self, x_bound, x_label, x_feat, x_k, x_b):
        with tf.variable_scope('G'):

            # 融合
            x_feat_act = tf.add(tf.multiply(x_feat, x_k), x_b)
            x_concat = tf.concat([x_bound, x_label, x_feat_act], 3)

            #
            input_downsampled = tf.nn.avg_pool(x_concat, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

            # G1
            _, G1_relu_up4 = G_base('G1', input_downsampled, self.batch)

            # G2_1
            G2_1_conv1 = conv('G2_1_conv1', x_concat, 7 * 7, 64, 1, None, True)
            G2_1_ins1 = ins_norm('G2_1_ins1', G2_1_conv1)
            G2_1_relu1 = relu('G2_1_relu1', G2_1_ins1)

            G2_1_conv2 = conv('G2_1_conv2', G2_1_relu1, 3 * 3, 128, 2, 1, False)
            G2_1_ins2 = ins_norm('G2_1_ins2', G2_1_conv2)
            G2_1_relu2 = relu('G2_1_relu2', G2_1_ins2)

            # 融合G1的输出和G2_1的输出 128
            G_add = tf.add(G1_relu_up4, G2_1_relu2, name='G_Add')

            # G2_2
            # res_block
            for i in range(3):
                name = 'G2_2_res'+str(i+1)
                G_add = res_block(name, G_add, channels=128)
            #
            G2_2_trans = conv_trans('G2_2_trans', G_add, 3*3, 64, 2, self.batch, True)
            G2_2_ins2 = ins_norm('G2_2_ins2', G2_2_trans)
            G2_2_relu2 = relu('G2_2_relu2', G2_2_ins2)
            
            # final convolution
            G2_2_conv_end = conv('G2_2_conv_end', G2_2_relu2, 7 * 7, 3, 1, None, True)
            G2_2_tanh_end = tanh('G2_2_tanh_end', G2_2_conv_end)

            return G2_2_tanh_end
        
    def build_D1(self, im, label, reuse):
        with tf.variable_scope('D1', reuse=reuse):
            x_ = tf.concat([im, label], 3)
            D = D_base('D', x_)
            return D

    def build_D2(self, im, label, reuse):
        with tf.variable_scope('D2', reuse=reuse):
            x_ = tf.concat([im, label], 3)
            x_pool = pool('pool_D', x_)
            D = D_base('D', x_pool)
            return D
        
    def encoder(self, x):
        with tf.variable_scope('Encoder'):
            x_encode, _ = G_base('encode', x, self.batch)
            return x_encode
        
    def forward(self):
        self.x_feat = self.encoder(self.real_im_)
        
        self.fake_im = self.build_G(self.bound_, self.onehot, self.x_feat, self.k, self.b)
        self.real_D1_out = self.build_D1(self.real_im_, self.onehot, False)

        self.fake_D1_out = self.build_D1(self.fake_im, self.onehot, True)

        self.real_D2_out = self.build_D2(self.real_im_, self.onehot, False)
        self.fake_D2_out = self.build_D2(self.fake_im, self.onehot, True)
       
    def cacu_loss(self):
        self.lsgan_d1 = tf.reduce_mean(0.5*tf.square(self.real_D1_out[-1]-1) + 0.5*tf.square(self.fake_D1_out[-1]))                    
        self.lsgan_d2 = tf.reduce_mean(0.5*tf.square(self.real_D2_out[-1]-1) + 0.5*tf.square(self.fake_D2_out[-1]))
        self.lsgan_g = 0.5*tf.reduce_mean(tf.square(self.fake_D2_out[-1]-1)) + 0.5*tf.reduce_mean(tf.square(self.fake_D1_out[-1]-1))
        self.feat_loss = feat_loss(self.real_D1_out, self.fake_D1_out, self.real_D2_out, self.fake_D2_out, self.feat_weight, self.d_weight)
        tf.summary.scalar('d1_loss', self.lsgan_d1)
        tf.summary.scalar('d2_loss', self.lsgan_d2)
        tf.summary.scalar('g_loss', self.lsgan_g)
        tf.summary.scalar('feat_loss', self.feat_loss)
        
    def train(self):
        lr = self.old_lr
        self.forward()
        self.cacu_loss()
        D1_vars = [var for var in tf.all_variables() if 'D1' in var.name]
        D2_vars = [var for var in tf.all_variables() if 'D2' in var.name]
        G_vars = [var for var in tf.all_variables() if 'G' in var.name]
        encoder_vars = [var for var in tf.all_variables() if 'Encoder' in var.name]

        optim_D1 = tf.train.AdamOptimizer(lr).minimize(self.lsgan_d1, var_list=D1_vars)
        optim_D2 = tf.train.AdamOptimizer(lr).minimize(self.lsgan_d2, var_list=D2_vars)
        optim_G_ALL = tf.train.AdamOptimizer(lr).minimize(self.lsgan_g+self.feat_loss, var_list=G_vars+encoder_vars)
        
        im_l, im_re, im_bound = self.read_and_decode(self.tf_record_dir)
        label_batch, real_batch, bound_batch = tf.train.batch([im_l, im_re, im_bound], batch_size=self.batch,
                                                              capacity=self.batch)
        
        with tf.Session() as sess:
            k_fed = np.ones([1], np.float32)
            b_fed = np.zeros([self.batch, self.im_high, self.im_width, 3], np.float32)
            
            sess.run(tf.global_variables_initializer())
            print("初始化变量完成")

            merge = tf.summary.merge_all()
            graph = tf.summary.FileWriter(self.log_dir, sess.graph)

            Saver = tf.train.Saver(max_to_keep=5)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            for ep in range(self.epoch):
                for j in range(self.n_im//self.batch):                    

                    label_fed, real_im_fed, bound_fed = sess.run([label_batch, real_batch, bound_batch])
                    print("加载数据完成")
                    dict_ = {self.label: label_fed, self.bound: bound_fed, self.real_im: real_im_fed, self.k: k_fed, self.b: b_fed}
                    step = int(ep*(self.n_im//self.batch)+j)

                    _, _ = sess.run([optim_D1, optim_D2], feed_dict=dict_)
                    print("优化D完成")

                    _, fake_im, Merge = sess.run([optim_G_ALL, self.fake_im, merge], feed_dict=dict_)
                    print("优化G完成")

                    graph.add_summary(Merge, step)

                    if (ep*self.n_im+j*self.batch)//self.save_iter == 0:
                        Save_im(fake_im, self.save_im_dir, ep, j)

                    if (j*self.batch+ep*self.n_im) % self.sace_ckpt_iter == 0:
                        num_trained = int(j*self.batch+ep*self.n_im)
                        Saver.save(sess, self.save_path+'/'+'model.ckpt', num_trained)
                        print('save success at num images trained: ', num_trained)

                if ep > self.decay_ep:

                    lr = self.old_lr - ep/self.decay_weight

            coord.request_stop()
            coord.join(threads)
            return True
        
    def Load_model(self, b_fed):
        #  b_fed is a feature vector extracted from the encoder's encoding and needs to be specified 
        #   by human (by clustering the results of the trained encoder).
        self.x_feat = self.encoder(self.real_im_)
        self.fake_im = self.build_G(self.bound_,self.onehot,self.x_feat,self.k,self.b)       
        G_vars = [var for var in tf.all_variables() if 'G' in var.name]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            graph = tf.summary.FileWriter(self.log_dir, sess.graph)
            Saver = tf.train.Saver(var_list=G_vars)
            Saver.restore(sess,self.ckpt_dir)
            
            label_fed,bound_fed = load_data(self.label_dir,self.inst_dir)
            #  k_fed must be zero, which means that the actual output of the encoder is not considered, because there is no ideal result color map when used. 
            #      (The characteristic input of G is: output(encoder)*k+b, k=1 during training, b=0)
            k_fed = np.zeros([1],np.float32)
            
            real_im_fed = np.zeros([np.shape(label_fed)[0],self.im_width,self.im_high,3],np.float32)
            
            dict_ = {self.label:label_fed,self.bound:bound_fed,self.real_im:real_im_fed,self.k:k_fed,self.b:b_fed}
            
            ims = sess.run(self.fake_im,feed_dict=dict_)
            Save_im(ims,self.save_im_dir,0,0)
            print(np.shape(ims))
