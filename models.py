import torch
import torch.nn as nn
from torch.autograd import Variable


class NCM(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, hidden_size, num_layers):
        super(NCM, self).__init__()
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder = Encoder(input_vocab_size, hidden_size, num_layers)
        self.decoder = Decoder(output_vocab_size, hidden_size, num_layers)


class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.cell = nn.LSTM(hidden_size, hidden_size, self.num_layers, batch_first=True)

    def forward(self, input_seqs):
        input_seqs = self.embedding(input_seqs)

        batch_size = input_seqs.size()[0]
        hidden_state = init_hidden(self.num_layers, batch_size, self.hidden_size)
        cell_state = init_hidden(self.num_layers, batch_size, self.hidden_size)
        thought = (hidden_state, cell_state)

        output, thought = self.cell(input_seqs, thought)
        return output, thought


class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.cell = nn.LSTM(hidden_size, hidden_size, self.num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax()

    def softmax_batch(self, linear_output):
        result = Variable(torch.zeros(linear_output.size())).cuda()
        batch_size = linear_output.size()[0]
        for i in xrange(batch_size):
            result[i] = self.softmax(linear_output[i])
        return result

    def forward(self, target_seqs, thought):
        target_seqs = self.embedding(target_seqs)
        output, thought = self.cell(target_seqs, thought)
        output = self.softmax_batch(self.out(output))
        return output, thought


def init_hidden(num_layers, batch_size, hidden_size):
    return Variable(torch.zeros(num_layers, batch_size, hidden_size)).cuda()
