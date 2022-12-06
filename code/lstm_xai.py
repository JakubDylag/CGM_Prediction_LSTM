from scipy.special import expit as sigmoid
import numpy as np

#https://www.gregcondit.com/articles/lstm-ref-card#:~:text=to%20make%20predictions-,Functions,-for%20an%20LSTM

def model_output(lstm_output, fc_Weight, fc_Bias):
    '''Takes the LSTM output and transforms it to our desired
    output size using a final, fully connected layer'''
    return np.dot(fc_Weight, lstm_output) + fc_Bias

class lstm_layer():
    def __init__(self, hidden_size, state, layer):
        self.hidden_size = hidden_size

        self.h = np.zeros(hidden_size)
        self.c = np.zeros(hidden_size)

        # Event (x) Weights and Biases for all gates
        self.Weights_xi = state['lstm1.weight_ih_l' + str(layer)][0:hidden_size].numpy()  # shape  [h, x]
        self.Weights_xf = state['lstm1.weight_ih_l' + str(layer)][hidden_size:hidden_size * 2].numpy()  # shape  [h, x]
        self.Weights_xl = state['lstm1.weight_ih_l' + str(layer)][
                          hidden_size * 2:hidden_size * 3].numpy()  # shape  [h, x]
        self.Weights_xo = state['lstm1.weight_ih_l' + str(layer)][
                          hidden_size * 3:hidden_size * 4].numpy()  # shape  [h, x]

        self.Bias_xi = state['lstm1.bias_ih_l' + str(layer)][0:hidden_size].numpy()  # shape is [h, 1]
        self.Bias_xf = state['lstm1.bias_ih_l' + str(layer)][hidden_size:hidden_size * 2].numpy()  # shape is [h, 1]
        self.Bias_xl = state['lstm1.bias_ih_l' + str(layer)][hidden_size * 2:hidden_size * 3].numpy()  # shape is [h, 1]
        self.Bias_xo = state['lstm1.bias_ih_l' + str(layer)][hidden_size * 3:hidden_size * 4].numpy()  # shape is [h, 1]

        # Hidden state (h) Weights and Biases for all gates
        self.Weights_hi = state['lstm1.weight_hh_l' + str(layer)][0:hidden_size].numpy()  # shape is [h, h]
        self.Weights_hf = state['lstm1.weight_hh_l' + str(layer)][
                          hidden_size:hidden_size * 2].numpy()  # shape is [h, h]
        self.Weights_hl = state['lstm1.weight_hh_l' + str(layer)][
                          hidden_size * 2:hidden_size * 3].numpy()  # shape is [h, h]
        self.Weights_ho = state['lstm1.weight_hh_l' + str(layer)][
                          hidden_size * 3:hidden_size * 4].numpy()  # shape is [h, h]

        self.Bias_hi = state['lstm1.bias_hh_l' + str(layer)][0:hidden_size].numpy()  # shape is [h, 1]
        self.Bias_hf = state['lstm1.bias_hh_l' + str(layer)][hidden_size:hidden_size * 2].numpy()  # shape is [h, 1]
        self.Bias_hl = state['lstm1.bias_hh_l' + str(layer)][hidden_size * 2:hidden_size * 3].numpy()  # shape is [h, 1]
        self.Bias_ho = state['lstm1.bias_hh_l' + str(layer)][hidden_size * 3:hidden_size * 4].numpy()  # shape is [h, 1]

        # Final, fully connected layer Weights and Bias
        self.fc_Weight = state['fc.weight'][0].numpy()  # shape is [h, output_size]
        self.fc_Bias = state['fc.bias'][0].numpy()  # shape is [,output_size]

    def forward(self, eventx, h):
        # print("h", h.shape)
        # print("c", self.c.shape)
        fg, f = self.forget_gate(eventx, h, self.Weights_hf, self.Bias_hf, self.Weights_xf, self.Bias_xf, self.c)
        # print("f", f.shape)
        # print("fg", fg.shape)
        ig, i = self.input_gate(eventx, h, self.Weights_hi, self.Bias_hi, self.Weights_xi, self.Bias_xi,
                                self.Weights_hl, self.Bias_hl, self.Weights_xl, self.Bias_xl)
        # print("i", i.shape)

        self.c = self.cell_state(f, i)
        # print("new c", self.c.shape)

        hg, self.h = self.output_gate(eventx, h, self.Weights_ho, self.Bias_ho, self.Weights_xo, self.Bias_xo,
                                      self.c)  # TODO: fix turns h to matrix
        # print("new h", self.h.shape)

        # gates.append((fg, ig , self.c, hg))
        # return [model_output(self.h, self.fc_Weight, self.fc_Bias), 0]
        return self.h, self.c, fg

    def forget_gate(self, x, h, Weights_hf, Bias_hf, Weights_xf, Bias_xf, prev_cell_state):
        # print("---" * 5)
        # print("x", x)
        # print("h", h.shape)

        # print("w_hf", Weights_hf.shape)
        forget_hidden = np.dot(Weights_hf, h) + Bias_hf
        # print("fh", forget_hidden.shape)

        # print("w_xf", Weights_xf.shape)
        forget_eventx = np.dot(Weights_xf, x) + Bias_xf
        # print("fx", forget_eventx.shape)
        gate = sigmoid(forget_hidden + forget_eventx)
        return gate, np.multiply(gate, prev_cell_state)

    def input_gate(self, x, h, Weights_hi, Bias_hi, Weights_xi, Bias_xi, Weights_hl, Bias_hl, Weights_xl, Bias_xl):
        ignore_hidden = np.dot(Weights_hi, h) + Bias_hi
        ignore_eventx = np.dot(Weights_xi, x) + Bias_xi
        learn_hidden = np.dot(Weights_hl, h) + Bias_hl
        learn_eventx = np.dot(Weights_xl, x) + Bias_xl
        gate = sigmoid(ignore_eventx + ignore_hidden)
        return gate, np.multiply(gate, np.tanh(learn_eventx + learn_hidden))

    def cell_state(self, forget_gate_output, input_gate_output):
        return forget_gate_output + input_gate_output

    def output_gate(self, x, h, Weights_ho, Bias_ho, Weights_xo, Bias_xo, cell_state):
        out_hidden = np.dot(Weights_ho, h) + Bias_ho
        out_eventx = np.dot(Weights_xo, x) + Bias_xo
        gate = sigmoid(out_eventx + out_hidden)
        return gate, np.multiply(gate, np.tanh(cell_state))
