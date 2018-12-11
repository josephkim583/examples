import torch.nn as nn
import torch
import csv
import data
import model
import operator

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, embedding, corpus, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        print ("EMBL ", embedding)
        file = csv.reader(open(embedding))
        embedding_dict = {row[0]: row[1:] for row in file if row and row[0]}

        # Get key with max value
        lengthcount = {}
        for key,value in embedding_dict.items():
            if (len(value) in lengthcount):
                lengthcount[len(value)] += 1
            else:
                lengthcount[len(value)] = 1

        ninp = max(lengthcount.items(), key=operator.itemgetter(1))[0]

        self.encoder = nn.Embedding(ntoken, ninp)

        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        
        #Goes through the embedding set, 
        #looks if any of the words show up in the corpus, 
        #and if it does show up, update the encoder vector values.
            

        self.init_weights()

        #Update embedding weights
        for key, value in embedding_dict.items():
            if key in corpus.dictionary.word2idx: 
                if (len(value) == ninp):
                    value = self.convert_to_int(value) 
                    vector = torch.FloatTensor([value])
                    self.encoder.weight.data[corpus.dictionary.word2idx[key]] = vector

        self.decoder.bias.data.zero_()
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def convert_to_int(self, string_list):
        for key,value in enumerate(string_list):
            string_list[key] = float(value)
        return (string_list)

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        # print (input)
        # print (self.encoder(input))
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

    