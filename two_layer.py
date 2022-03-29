import json
import numpy
import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse


def cross_entropy(pred, label):
    shifted_logits = pred - np.max(pred, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = pred.shape[0]
    loss = -np.sum(log_probs[np.arange(N), label]) / N
    x_grad = probs.copy()
    x_grad[np.arange(N), label] -= 1
    x_grad /= N
    return loss, x_grad


class TwoLayerNN:
    def __init__(self, lr=0.001, num_in=784, num_out=10, hidden=None, weight_scale=1e-1,
                 L2=1e-4, epoch=5, batch_size=100, visualize=False, checkpoint=None):
        if hidden is None:
            hidden = [100, 64]
        self.lr = lr
        self.num_in = num_in
        self.num_out = num_out
        self.params = {}
        self.hidden = hidden
        self.weight_scale = weight_scale
        self.L2 = L2
        self.epoch = epoch
        self.batch_size = batch_size
        self.visualize = visualize
        self.checkpoint = checkpoint
        self.loss = []
        self.acc = []
        self.test_loss = []
        self.test_acc = []
        self.decay = 0.999

        self.init_weights()

    def init_weights(self):
        self.params['W1'] = np.random.randn(self.num_in, self.hidden[0]) * self.weight_scale
        self.params['W2'] = np.random.randn(self.hidden[0], self.hidden[1]) * self.weight_scale
        self.params['W3'] = np.random.randn(self.hidden[1], self.num_out) * self.weight_scale
        self.params['b1'] = np.zeros(self.hidden[0], )
        self.params['b2'] = np.zeros(self.hidden[1], )
        self.params['b3'] = np.zeros(self.num_out, )

    def train(self, input, y):
        for i in range(self.epoch):
            loss = 0
            sum_acc = 0
            for batch_ in range(int(len(input) / self.batch_size)):
                batch_input = input[self.batch_size * batch_:(batch_ + 1) * self.batch_size]

                # forward
                hidden1_ = np.dot(batch_input, self.params['W1']) + self.params['b1']
                hidden1 = sigmoid(hidden1_)

                hidden2_ = np.dot(hidden1, self.params['W2']) + self.params['b2']
                hidden2 = sigmoid(hidden2_)

                output = np.dot(hidden2, self.params['W3']) + self.params['b3']

                # compute loss
                loss_, loss_grad = cross_entropy(output,
                                                 y[self.batch_size * batch_:(batch_ + 1) * self.batch_size])
                loss = loss_ + 0.5 * self.L2 * (
                        np.sum(np.square(self.params['W3'])) + np.sum(np.square(self.params['W2'])) + np.sum(
                    np.square(self.params['W1'])))

                # compute the accuracy
                y_pred = np.argmax(output, axis=1).reshape(1, -1)
                y_true = y[self.batch_size * batch_:(batch_ + 1) * self.batch_size].reshape(1, -1)
                sum_ = 0.0
                # print(y_pred.shape[1], split)
                for c in range(y_pred.shape[1]):
                    if y_pred[0, c] == y_true[0, c]:
                        sum_ = sum_ + 1
                sum_acc += sum_

                # back pro
                w3_grad = np.dot(hidden2.T, loss_grad)
                b3_grad = np.sum(loss_grad, axis=0)
                hidden2_grad = np.dot(loss_grad, self.params['W3'].T) * (1 - hidden2) * hidden2

                w2_grad = np.dot(hidden1.T, hidden2_grad)
                b2_grad = np.sum(hidden2_grad, axis=0)
                hidden1_grad = np.dot(hidden2_grad, self.params['W2'].T) * (1 - hidden1) * hidden1

                w1_grad = np.dot(batch_input.T, hidden1_grad)  # 2*4 and 4*2 => 2*2
                b1_grad = np.sum(hidden1_grad, axis=0)  # 1 * 2

                # L2
                w3_grad += self.params['W3'] * self.L2
                w2_grad += self.params['W2'] * self.L2
                w1_grad += self.params['W1'] * self.L2

                # backward
                self.params['W3'] -= self.lr * w3_grad
                self.params['b3'] -= self.lr * b3_grad
                self.params['W2'] -= self.lr * w2_grad
                self.params['b2'] -= self.lr * b2_grad
                self.params['W1'] -= self.lr * w1_grad
                self.params['b1'] -= self.lr * b1_grad

            # print(sum_acc, len(input))
            self.lr = self.lr * self.decay
            self.loss.append(loss)
            self.acc.append(sum_acc * 100 / len(input))
            self.test(test_x, test_label)
            if i % 10 == 0:
                print('Epochs {} -- Acc: [{:.3f}%], Loss: [{:.5f}]'.format(i, sum_acc * 100 / len(input), loss))

        print(f'保存模型至{self.checkpoint}')
        with open(self.checkpoint, 'w', encoding='UTF-8') as f:
            output_data = {}
            for par in self.params:
                output_data[par] = self.params[par].tolist()
            json.dump(output_data, f)

    def test(self, input, y, load_checkpoint=None):
        if load_checkpoint:
            with open(load_checkpoint, 'r', encoding='UTF-8') as f:
                params = json.load(f)
            for par in self.params:
                self.params[par] = np.array(params[par])

        hidden1_ = np.dot(input, self.params['W1']) + self.params['b1']
        if self.visualize:
            res = numpy.uint8(hidden1_[0] * 255 + 255 / 2).reshape(10, 10)
            plt.title('first layer parameters')
            plt.imshow(res, cmap=plt.cm.gray)
            plt.show()

        hidden1 = sigmoid(hidden1_)

        hidden2_ = np.dot(hidden1, self.params['W2']) + self.params['b2']
        if self.visualize:
            res = numpy.uint8(hidden2_[0] * 255 + 255 / 2).reshape(8, 8)
            plt.title('second layer parameters')
            plt.imshow(res, cmap=plt.cm.gray)
            plt.show()

        hidden2 = sigmoid(hidden2_)

        output_ = np.dot(hidden2, self.params['W3']) + self.params['b3']
        output = sigmoid(output_)

        # compute test loss without l2 normalization
        test_loss, dout = cross_entropy(output, y)
        test_loss += 0.5 * self.L2 * (
                np.sum(np.square(self.params['W3'])) + np.sum(np.square(self.params['W2'])) + np.sum(
            np.square(self.params['W1'])))

        self.test_loss.append(test_loss)
        # compute the accuracy
        y_pred = np.argmax(output, axis=1).reshape(1, -1)
        y_true = y.reshape(1, -1)
        sum_ = 0.0
        for c in range(y_pred.shape[1]):
            if y_pred[0, c] == y_true[0, c]:
                sum_ = sum_ + 1

        self.test_acc.append(100.0 * sum_ / len(input))
        print('Test acc is {:.5f}'.format(100.0 * sum_ / len(input)))
        return 100.0 * sum_ / len(input)

    def get_train_loss(self):
        return self.loss

    def get_train_acc(self):
        return self.acc

    def get_test_loss(self):
        return self.test_loss

    def get_test_acc(self):
        return self.test_acc

    def visualize_parameters(self):
        self.visualize = True


def load_data(train_file='mnist_train.csv', test_file='mnist_test.csv'):
    train = csv.reader(open(train_file, 'r'))
    train_content = []
    for index, line in enumerate(train):
        if index == 0:
            continue
        train_content.append(line)
    test = csv.reader(open(test_file, 'r'))
    test_content = []
    for index, line in enumerate(test):
        if index == 0:
            continue
        test_content.append(line)

    train_content = np.array(train_content, dtype=np.float32)
    test_content = np.array(test_content, dtype=np.float32)

    train_label = np.array(train_content[:, 0], dtype=int)
    train_x = train_content[:, 1:]
    test_label = np.array(test_content[:, 0], dtype=int)
    test_x = test_content[:, 1:]

    # normalization
    train_x = (train_x - 255 / 2) / 255
    test_x = (test_x - 255 / 2) / 255

    return train_x, test_x, train_label, test_label


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def train_model():
    model.train(train_x, train_label)


def load_checkpoint_and_test(checkpoint):
    model.test(test_x, test_label, checkpoint)


if __name__ == '__main__':
    train_x, test_x, train_label, test_label = load_data()
    parser = argparse.ArgumentParser(description="Parameters Parse")
    parser.add_argument('--lr', default=0.5, type=float)
    parser.add_argument('--hidden', default=[100, 64])
    parser.add_argument('--L2', default=1e-5)
    parser.add_argument('--epoch', default=500)
    parser.add_argument('--batch_size', default=1000)
    parser.add_argument('--is_test', default=False)
    parser.add_argument('--checkpoint', default='params.json')
    parser.add_argument('--visualize', default=False)
    args = parser.parse_args()

    model = TwoLayerNN(lr=args.lr, hidden=args.hidden, L2=args.L2, epoch=args.epoch, batch_size=args.batch_size,
                       visualize=args.visualize, checkpoint=args.checkpoint)

    x = range(args.epoch)
    if not args.is_test:
        print('训练模型中...')
        train_model()
        print(f'模型训练结束，参数保存至{args.checkpoint}')
        loss = model.get_train_loss()
        test_loss = model.get_test_loss()
        plt.plot(x, loss)
        plt.plot(x, test_loss)
        plt.legend(['train loss', 'test loss'])
        plt.title('Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()

        test_acc = model.get_test_acc()
        plt.plot(x, test_acc)
        plt.title('Test Accuracy Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.show()
    else:
        print(f'读取模型参数，参数路径为{args.checkpoint}')
        print('测试模型中...')
        if args.visualize:
            model.visualize_parameters()
        load_checkpoint_and_test(args.checkpoint)
    # 参数搜索
    # all_result = []
    # for hidden in [[[225, 100], [100, 64], [225, 64], [169, 64]]]:
    #     for L2 in [1e-5, 5e-5, 1e-4]:
    #         for lr in [0.5, 0.1, 0.01]:
    #             model.train(train_x, train_label)
    #             acc = model.test(test_x, test_label)
    #             all_result.append(acc)
    # best = np.argmax(np.array(all_result))

