import tensorflow as tf
import constants as const
import glob


class FltRPN:
    def make_data(self, fns):
        data = tf.data.TFRecordDataset(fns, compression_type='GZIP')
        data = data.map(self.decode, num_parallel_calls=8)
        # print("Disables shuffle buffer")
        data = data.shuffle(256)
        data = data.repeat()
        data = data.batch(const.BS)
        data = data.prefetch(4)

        return data

    def decode(self, example):
        stuff = tf.parse_single_example(example, features={
            'images': tf.FixedLenFeature([], tf.string),
            'depths': tf.FixedLenFeature([], tf.string),
            'bboxes': tf.FixedLenFeature([], tf.string),
            'pos_equal_one': tf.FixedLenFeature([], tf.string),
            'neg_equal_one': tf.FixedLenFeature([], tf.string),
            'anchor_reg': tf.FixedLenFeature([], tf.string),
            'num_obj': tf.FixedLenFeature([], tf.string),
            'voxel': tf.FixedLenFeature([], tf.string),
            'voxel_obj': tf.FixedLenFeature([], tf.string)
        })

        images = tf.decode_raw(stuff['images'], tf.float64)
        images = tf.reshape(images, (const.N, const.resolution, const.resolution, 3))
        pos_equal_one = tf.cast(tf.decode_raw(stuff['pos_equal_one'], tf.int64), tf.float64)
        pos_equal_one = tf.reshape(pos_equal_one, (32, 32))
        anchor_reg = tf.decode_raw(stuff['anchor_reg'], tf.float64)
        anchor_reg = tf.reshape(anchor_reg, (32, 32, 6))
        return images, pos_equal_one, anchor_reg

    def first_layers(self, data_0, data_90):
        # Image from -90 degree
        with tf.name_scope('projection_0'):
            x_0 = tf.layers.conv2d(data_0, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
            x_0 = tf.layers.conv2d(x_0, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
            x_0 = tf.layers.max_pooling2d(x_0, pool_size=(2, 2), strides=(2, 2), padding='same')

            # Image from 0 degree
        with tf.name_scope('projection_90'):
            x_90 = tf.layers.conv2d(data_90, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
            x_90 = tf.layers.conv2d(x_90, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
            x_90 = tf.layers.max_pooling2d(x_90, pool_size=(2, 2), strides=(2, 2), padding='same')

        with tf.name_scope('merge_3D'):
            # x_0[z, x]
            # x_90[z, y]
            FT = tf.tile(x_0[:, :, :, None, :], [1, 1, 1, 64, 1])
            FT = FT + tf.tile(x_90[:, :, None, :, :], [1, 1, 64, 1, 1])

        tf.summary.image('data_0', data_0)
        tf.summary.image('data_90', data_90)

        return FT

    def rpn(self, fl_input):
        # fl_input[batch, z, x, y, c]
        with tf.name_scope('rpn'):
            with tf.name_scope('conv3D'):
                temp_conv = tf.layers.conv3d(fl_input, 128, 3, strides=(2, 1, 1), activation=tf.nn.relu, padding="same")
                temp_conv = tf.layers.conv3d(temp_conv, 64, 3, strides=(1, 1, 1), activation=tf.nn.relu, padding="same")
                temp_conv = tf.layers.conv3d(temp_conv, 64, 3, strides=(2, 1, 1), activation=tf.nn.relu, padding="same")

                temp_conv = tf.transpose(temp_conv, perm=[0, 2, 3, 4, 1])
                temp_conv = tf.reshape(temp_conv, [-1, temp_conv.shape[1], temp_conv.shape[2], (temp_conv.shape[3]*temp_conv.shape[4])])
            with tf.name_scope('block1'):
                # block1:
                temp_conv = tf.layers.conv2d(temp_conv, 128, 3, strides=(2, 2), activation=tf.nn.relu, padding="same")
                temp_conv = tf.layers.conv2d(temp_conv, 128, 3, strides=(1, 1), activation=tf.nn.relu, padding="same")
                temp_conv = tf.layers.conv2d(temp_conv, 128, 3, strides=(1, 1), activation=tf.nn.relu, padding="same")
                temp_conv = tf.layers.conv2d(temp_conv, 128, 3, strides=(1, 1), activation=tf.nn.relu, padding="same")
                deconv1 = tf.layers.conv2d_transpose(temp_conv, 256, 3, strides=(1, 1), padding="same")
            with tf.name_scope('block2'):
                # block2:
                temp_conv = tf.layers.conv2d(temp_conv, 128, 3, strides=(2, 2), activation=tf.nn.relu, padding="same")
                temp_conv = tf.layers.conv2d(temp_conv, 128, 3, strides=(1, 1), activation=tf.nn.relu, padding="same")
                temp_conv = tf.layers.conv2d(temp_conv, 128, 3, strides=(1, 1), activation=tf.nn.relu, padding="same")
                temp_conv = tf.layers.conv2d(temp_conv, 128, 3, strides=(1, 1), activation=tf.nn.relu, padding="same")
                temp_conv = tf.layers.conv2d(temp_conv, 128, 3, strides=(1, 1), activation=tf.nn.relu, padding="same")
                temp_conv = tf.layers.conv2d(temp_conv, 128, 3, strides=(1, 1), activation=tf.nn.relu, padding="same")
                deconv2 = tf.layers.conv2d_transpose(temp_conv, 256, 2, strides=(2, 2), padding="same")
            with tf.name_scope('block3'):
                # block3:
                temp_conv = tf.layers.conv2d(temp_conv, 256, 3, strides=(2, 2), activation=tf.nn.relu, padding="same")
                temp_conv = tf.layers.conv2d(temp_conv, 256, 3, strides=(1, 1), activation=tf.nn.relu, padding="same")
                temp_conv = tf.layers.conv2d(temp_conv, 256, 3, strides=(1, 1), activation=tf.nn.relu, padding="same")
                temp_conv = tf.layers.conv2d(temp_conv, 256, 3, strides=(1, 1), activation=tf.nn.relu, padding="same")
                temp_conv = tf.layers.conv2d(temp_conv, 256, 3, strides=(1, 1), activation=tf.nn.relu, padding="same")
                temp_conv = tf.layers.conv2d(temp_conv, 256, 3, strides=(1, 1), activation=tf.nn.relu, padding="same")
                deconv3 = tf.layers.conv2d_transpose(temp_conv, 256, 4, strides=(4, 4), padding="SAME")

            # final:
            temp_conv = tf.concat([deconv3, deconv2, deconv1], -1)
            p_map = tf.layers.conv2d(temp_conv, 1, 1, strides=(1, 1), activation=tf.nn.relu, padding="valid")
            r_map = tf.layers.conv2d(temp_conv, 6, 1, strides=(1, 1), activation=tf.nn.relu, padding="valid")
            p_pos = tf.sigmoid(p_map)
            self.summary_p_pos = tf.summary.image('p_pos', tf.expand_dims(p_pos[0], axis=0))
            p_pos = tf.reshape(p_pos, (-1, 32, 32))

        return p_pos, r_map

    def smooth_l1(self, deltas, targets, sigma=3.0):
        sigma2 = sigma * sigma
        diffs = tf.subtract(deltas, targets)
        smooth_l1_signs = tf.cast(tf.less(tf.abs(diffs), 1.0 / sigma2), tf.float32)

        smooth_l1_option1 = tf.multiply(diffs, diffs) * 0.5 * sigma2
        smooth_l1_option2 = tf.abs(diffs) - 0.5 / sigma2
        smooth_l1_add = tf.multiply(smooth_l1_option1, smooth_l1_signs) + \
            tf.multiply(smooth_l1_option2, 1 - smooth_l1_signs)
        smooth_l1 = smooth_l1_add

        return smooth_l1

    def calc_loss(self, p_pos, r_map, pos_equal_one, anchors_reg):
        pos_equal_one_sum = tf.reduce_sum(pos_equal_one, axis=[1, 2])
        neg_equal_one_sum = tf.reduce_sum(1 - pos_equal_one, axis=[1, 2])
        cls_pos_loss = (-pos_equal_one * tf.log(p_pos + 1e-6)) / tf.reshape(pos_equal_one_sum, [-1, 1, 1])
        cls_neg_loss = (-(1 - pos_equal_one) * tf.log(1 - p_pos + 1e-6)) / tf.reshape(neg_equal_one_sum, [-1, 1, 1])
        loss_prob = tf.reduce_sum(cls_pos_loss + cls_neg_loss) / const.BS

        pos_equal_one_expanded = tf.expand_dims(pos_equal_one, 3)
        # r_map_mask = tf.tile(pos_equal_one_expanded, [1, 1, 1, 6])
        # loss_reg = tf.reduce_sum(self.smooth_l1(r_map * r_map_mask, anchors_reg * r_map_mask) / tf.reshape(pos_equal_one_sum, [-1, 1, 1, 1])) / const.BS
        #
        # loss = loss_prob + loss_reg
        self.summary_pos_equal_one = tf.summary.image('real_pos', tf.expand_dims(tf.expand_dims(pos_equal_one[0], 2), 0))
        # self.summary_loss = tf.summary.scalar('loss', loss)
        self.summary_loss_prob = tf.summary.scalar('loss_prob', loss_prob)
        # self.summary_loss_reg = tf.summary.scalar('loss_reg', loss_reg)

        return loss_prob

    def train_step(self, data):
        images, pos_equal_one, anchor_reg = data
        with tf.name_scope('train'):
            FT = self.first_layers(images[:, 0], images[:, 1])
            p_pos, r_map = self.rpn(FT)
            loss = self.calc_loss(p_pos, r_map, pos_equal_one, anchor_reg)
        with tf.name_scope('optimize'):
            opt = tf.train.AdamOptimizer(const.lr, const.mom).minimize(loss)

        return opt, loss

    def go(self):
        fns = sorted(glob.glob(const.TF_RECORD_DIR + '*.tfrecord'))
        print("Number of examples", len(fns))
        train_data = self.make_data(fns)
        val_data = self.make_data(fns)
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, train_data.output_types, train_data.output_shapes)
        next_element = iterator.get_next()

        training_iterator = train_data.make_one_shot_iterator()
        validation_iterator = val_data.make_one_shot_iterator()

        opt, loss = self.train_step(next_element)
        merged_summary = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('log/train')
        # merged_val = tf.summary.merge([self.summary_loss, self.summary_loss_prob, self.summary_loss_reg, self.summary_p_pos, self.summary_pos_equal_one])
        self.val_writer = tf.summary.FileWriter('log/val')

        with tf.Session() as sess:
            self.train_writer.add_graph(sess.graph)
            sess.run(tf.global_variables_initializer())
            training_handle = sess.run(training_iterator.string_handle())
            validation_handle = sess.run(validation_iterator.string_handle())
            for i in range(2000):
                s, _ = sess.run([merged_summary, opt], feed_dict={handle: training_handle})
                self.train_writer.add_summary(s, i)
                print(i)
                if i%const.valp == 0:
                    v, _ = sess.run([merged_summary, loss], feed_dict={handle: validation_handle})

                    self.val_writer.add_summary(v, i)


R = FltRPN()
R.go()
