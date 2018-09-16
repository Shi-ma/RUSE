import torch
import tensorflow as tf
import tensorflow_hub as hub
import json
import numpy as np
import sys
import os
import io

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('mode', 'test', 'train or dev or test')
tf.flags.DEFINE_string('tsv_path', None, 'input: tsv_path')
tf.flags.DEFINE_string('npz_out_dir', None, 'output file directory')
# tf.flags.DEFINE_integer('gpu_id', None, 'gpu_id')
tf.flags.DEFINE_string('sr_model', None, 'eg. IS or QT or USE')
tf.flags.DEFINE_string('case', 'true', 'true or lower')
tf.flags.DEFINE_string('is_dir', None, 'IS directory')
tf.flags.DEFINE_string('qt_dir', None, 'QT directory')
tf.flags.DEFINE_string('qt_pretrained_dir', None, 'QT pre-trained models directory include models and dictionaries')

sys.path.append(FLAGS.is_dir)
from models import InferSent
sys.path.append(os.path.join(FLAGS.qt_dir, 'src'))
import configuration
import encoder_manager



def load_model(FLAGS):
    if FLAGS.sr_model == 'IS':
        #Load InferSent
        MODEL_PATH = os.path.join(FLAGS.is_dir, 'encoder/infersent1.pkl')

        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                        'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}
        model = InferSent(params_model)
        model.load_state_dict(torch.load(MODEL_PATH))
        W2V_PATH = os.path.join(FLAGS.is_dir, 'dataset/GloVe/glove.840B.300d.txt')
        model.set_w2v_path(W2V_PATH)
    elif FLAGS.sr_model == 'QT':
        # Load Quick-Thought
        tf.flags.DEFINE_string("eval_task", "MSRP",
                               "Name of the evaluation task to run. Available tasks: "
                               "MR, CR, SUBJ, MPQA, SICK, MSRP, TREC.")
        tf.flags.DEFINE_string("data_dir", None, "Directory containing training data.")
        tf.flags.DEFINE_float("uniform_init_scale", 0.1, "Random init scale")
        tf.flags.DEFINE_integer("batch_size", 400, "Batch size")
        tf.flags.DEFINE_boolean("use_norm", False,
                                "Normalize sentence embeddings during evaluation")
        tf.flags.DEFINE_integer("sequence_length", 30, "Max sentence length considered")
        tf.flags.DEFINE_string("model_config", os.path.join(FLAGS.qt_dir, "model_configs/MC-UMBC/eval.json"), "Model configuration json")
        tf.flags.DEFINE_string("results_path", os.path.join(FLAGS.qt_pretrained_dir, "models"), "Model results path")
        tf.flags.DEFINE_string("Glove_path", os.path.join(FLAGS.qt_pretrained_dir, "dictionaries/GloVe"), "Path to Glove dictionary")
        tf.logging.set_verbosity(tf.logging.INFO)

        model = encoder_manager.EncoderManager()

        with open(FLAGS.model_config) as json_config_file:
            model_config = json.load(json_config_file)
        if type(model_config) is dict:
            model_config = [model_config]

        for mdl_cfg in model_config:
            model_config = configuration.model_config(mdl_cfg, mode="encode")
            model.load_model(model_config)
    elif FLAGS.sr_model == 'USE':
        model = hub.Module('https://tfhub.dev/google/universal-sentence-encoder-large/2')

    return model


# def length_check(txt):
#     if ',' in txt:
#         txt.replace(',', '')
#         txt_list = txt.split()
#     else:
#         txt_list = txt.split()[:100]
#     if len(txt_list) > 100:
#         txt_list = txt_list[:100]
#
#     return ' '.join(txt_list)



def tsv2npz(model, FLAGS):
    refs = list()
    outs = list()
    labels = list()

    for line in io.open(FLAGS.tsv_path, encoding='utf-8'):
        ref = line.split('\t')[0]
        out = line.split('\t')[1]
        if len(out.split()) == 0:
            out = '.'

        if FLAGS.case == 'true':
            refs.append(ref)
            outs.append(out)
        elif FLAGS.case == 'lower':
            refs.append(ref.lower())
            outs.append(out.lower())
        if FLAGS.mode == 'train' or FLAGS.mode == 'dev':
            labels.append(float(line.split('\t')[2]))

    if FLAGS.sr_model == 'IS':
        model.build_vocab(refs + outs, tokenize=True)
        ref_embs = model.encode(refs, tokenize=True)
        out_embs = model.encode(outs, tokenize=True)
    # elif FLAGS.sr_model == 'QT':
    #     ref_embs = model.encode(refs)
    #     out_embs = model.encode(outs)
    elif FLAGS.sr_model == 'QT':
        ref_embs = list()
        out_embs = list()
        for ref in refs:
            try:
                ref_embs.append(*(model.encode([ref])))
            except:
                ref_embs.append(np.zeros(4800))
        for out in outs:
            try:
                out_embs.append(*(model.encode([out])))
            except:
                out_embs.append(*(model.encode([out])))
    elif FLAGS.sr_model == 'USE':
        with tf.Session() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            ref_embs = session.run(model(refs))
        with tf.Session() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            out_embs = session.run(model(outs))
    features = [np.concatenate([u, v, abs(u - v), u * v], axis=0) for u, v in zip(ref_embs, out_embs)]

    return features, labels


if __name__ == '__main__':
    if not FLAGS.npz_out_dir:
        print('npz_out_dir not defined.')

    print('< load model ... >')
    model = load_model(FLAGS)
    print('\n< make npz ... >')
    features, labels = tsv2npz(model, FLAGS)
    # print(len(features[0]))

    print('\n< save npz ... >')
    npz_out_path = os.path.join(FLAGS.npz_out_dir, '{}.npz'.format(FLAGS.sr_model))
    np.savez(npz_out_path, features=features, labels=labels)

    print('\n <completed >')