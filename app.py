import pickle

from flask import Flask, request, jsonify
import torch
from model import EncoderRNN, AttnDecoderRNN, Lang, tensorFromSentence, MAX_LENGTH, SOS_token, EOS_token, \
    normalizeString

from flask import Flask
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('input_lang1.pkl', 'rb') as f:
    input_lang1 = pickle.load(f)

with open('input_lang2.pkl', 'rb') as f:
    input_lang2 = pickle.load(f)

with open('output_lang1.pkl', 'rb') as f:
    output_lang1 = pickle.load(f)

with open('output_lang2.pkl', 'rb') as f:
    output_lang2 = pickle.load(f)

def load_model1(encoder_path, decoder_path):
    encoder = EncoderRNN(input_lang1.n_words, 256).to(device)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    decoder = AttnDecoderRNN(256, output_lang1.n_words, dropout_p=0.1, max_length=MAX_LENGTH).to(device)
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    encoder.eval()
    decoder.eval()
    return encoder, decoder

def load_model2(encoder_path, decoder_path):
    encoder = EncoderRNN(input_lang2.n_words, 256).to(device)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    decoder = AttnDecoderRNN(256, output_lang2.n_words, dropout_p=0.1, max_length=MAX_LENGTH).to(device)
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    encoder.eval()
    decoder.eval()
    return encoder, decoder

def evaluate1(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang1, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == SOS_token:
                decoded_words.append('<SOS>')
                break
            else:
                decoded_words.append(output_lang1.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words

def evaluate2(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang2, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == SOS_token:
                decoded_words.append('<SOS>')
                break
            else:
                decoded_words.append(output_lang2.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words


@app.route('/translate1', methods=['POST'])
def translate1():
    encoder, decoder = load_model1('encoder1.pth', 'attn_decoder1.pth')
    data = request.get_json()
    sentence = normalizeString(data['sentence'])
    output_words = evaluate1(encoder, decoder, sentence)
    output_sentence = ' '.join(output_words)
    return jsonify({'translation': output_sentence})


@app.route('/translate2', methods=['POST'])
def translate2():
    encoder, decoder = load_model2('encoder2.pth', 'attn_decoder2.pth')
    data = request.get_json()
    sentence = normalizeString(data['sentence'])
    output_words = evaluate2(encoder, decoder, sentence)
    output_sentence = ' '.join(output_words)
    return jsonify({'translation': output_sentence})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)