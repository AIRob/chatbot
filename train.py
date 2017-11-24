import random
import time

import torch
from torch.autograd import Variable

from utils import pad_seqs


def train_iters(model, train, val, batch_size, optimizer, iters, print_iters):
    train_losses = []
    val_losses = []

    start_time = time.time()
    for i in xrange(1, iters + 1):
        train_batch = [random.choice(train) for _ in xrange(batch_size)]
        val_batch = [random.choice(val) for _ in xrange(batch_size)]

        train_loss = get_loss(model, train_batch)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        val_loss = get_loss(model, val_batch, inference_only=True)

        train_losses.append(train_loss.data[0])
        val_losses.append(val_loss.data[0])

        if i % print_iters == 0:
            end_time = time.time()

            avg_train_loss = sum(train_losses[-print_iters:]) / print_iters
            avg_val_loss = sum(val_losses[-print_iters:]) / print_iters

            epoch = (batch_size * i) / len(train)

            string = 'epoch: {}, iters: {}, train loss: {:.2f}, val loss: {:.2f}, time: {:.2f} s'
            print string.format(epoch, i, avg_train_loss, avg_val_loss, end_time - start_time)

            start_time = time.time()

    return train_losses, val_losses


def get_loss(model, batch, inference_only=False):
    answer_lens = [len(a) for _, a in batch]

    questions = pad_seqs([q for q, _ in batch])
    answers = pad_seqs([a for _, a in batch])

    questions = Variable(torch.LongTensor(questions), volatile=inference_only).cuda()
    answers = Variable(torch.LongTensor(answers), volatile=inference_only).cuda()

    # print questions.size(), answers.size()

    q_embedded = model.embedding(questions)
    a_embedded = model.embedding(answers)

    # print questions.size(), answers.size()
    _, thought = model.encoder(q_embedded)
    decoder_output, _ = model.decoder(a_embedded, thought)

    loss = 0
    loss_fn = torch.nn.NLLLoss()
    batch_size = len(batch)
    for i in xrange(batch_size):
        loss += loss_fn(decoder_output[i, :answer_lens[i] - 1], answers[i, 1:answer_lens[i]])

    return loss / batch_size
