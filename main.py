import argparse
import datetime

import numpy as np
import tensorflow as tf

from util.Dataset import Dataset
from models.CNN import Char_CNN
from models.MLP import MLP

def _batch_loader(iterable, n=1):
    length = len(iterable)
    for n_idx in range(0, length, n):
        yield iterable[n_idx:min(n_idx + n, length)]


def is_better_result(best, cur):
    return best < cur

def get_model(model):
    """
    Get Model instance
    """
    assert model in ['CNN', 'MLP']

    if model == 'CNN': return Char_CNN(config, fc_layers, filter_sizes)
    else: return MLP(config, fc_layers)

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument('--epochs', type=int, default=300)
    args.add_argument('--batch', type=int, default=256)
    args.add_argument('--lr', type=float, default=0.001)
    args.add_argument('--strmaxlen', type=int, default=150)
    args.add_argument('--charsize', type=int, default=300)
    args.add_argument('--filter_num', type=int, default=64)
    args.add_argument('--emb', type=int, default=128)
    args.add_argument('--eumjeol', type=bool, default=False)
    args.add_argument('--threshold', type=float, default=0.5)
    args.add_argument('--model', type=str, default='CNN')
    config = args.parse_args()
    if config.eumjeol:
        config.strmaxlen = 100
        config.charsize = 2510

    DATASET_PATH = './data/'
    DISPLAY_STEP = 30
    SUBTEST_STEP = 1

    filter_sizes = [3, 4]                   # Filter sizes for CNN
    fc_layers = [1500, 500, 100, 30]        # Dimensions of FC layers
    
    # Model specification
    ##############################################
    model = get_model(config.model)

    output_prob = model.output_prob
    train_step = model.train_step
    loss = model.loss
    x1 = model.x1
    x2 = model.x2
    y_ = model.y_

    ##############################################

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    print('=' * 15 + "MODEL INFO" + '=' * 15)
    print(config)

    # Loading Data
    print("Loading Dataset")
    TRAIN_SET = Dataset(dataset_path=DATASET_PATH, mode='train', max_len=config.strmaxlen, eumjeol=config.eumjeol)
    TEST_SET = Dataset(dataset_path=DATASET_PATH, mode='test', max_len=config.strmaxlen, eumjeol=config.eumjeol)
    dataset_len = len(TRAIN_SET)
    one_batch_size = dataset_len // config.batch
    if dataset_len % config.batch != 0:
        one_batch_size += 1

    # Train

    patience = 5  # Patience for early stop. Stop if no improvement has been made for patience epoch after best
    best_result = -1
    best_epoch = -1
    print('=' * 15 + "TRAINING START" + '=' * 15)
    for epoch in range(1, config.epochs + 1):
        avg_loss = 0.0
        for i, data in enumerate(_batch_loader(TRAIN_SET, config.batch)):
            data1 = data[0]
            data2 = data[1]
            labels = data[2].flatten()

            feed_dict = {x1: data1, x2: data2, y_: labels}

            _, l = sess.run([train_step, loss], feed_dict=feed_dict)

            # Print batch loss per specified step
            if i % DISPLAY_STEP == 0:
                time_str = datetime.datetime.now().isoformat()
                print('[%s] Batch : (%3d/%3d), LOSS in this minibatch : %.3f' % (
                time_str, i, one_batch_size, float(l)))
            avg_loss += float(l)
        print('Epoch: ', epoch, ' Train_Loss: ', float(avg_loss / one_batch_size))
        # Test validation set per specified step
        if epoch % SUBTEST_STEP == 0:
            print('\n' + '=' * 8 + "[Epoch %d] VALIDATION" % epoch + '=' * 8)
            res = []
            for i, data in enumerate(_batch_loader(TEST_SET, config.batch)):
                data1 = data[0]
                data2 = data[1]

                feed_dict = {x1: data1, x2: data2}
                temp_res = sess.run(output_prob, feed_dict=feed_dict)
                res += list(temp_res)
            pred = np.array(np.array(res) > config.threshold, dtype=np.int32)
            labels = TEST_SET.labels.flatten()
            correct = len(np.where(pred == labels)[0])
            accuracy = correct / len(TEST_SET)

            print("Accuracy : %.4f" % accuracy)

            if is_better_result(best_result, accuracy) or best_result == -1:
                best_result = accuracy
                best_epoch = epoch
            print("")