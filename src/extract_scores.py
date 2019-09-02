import argparse
from scipy.stats import pearsonr



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('metric', help='name of metric')
    parser.add_argument('lang_pair', help='lang_pair')
    args = parser.parse_args()

    src_lang, trg_lang = args.lang_pair.split('-')
    if args.lang_pair == 'cs-en':
        sys_list = ['online-A.0', 'online-B.0', 'PJATK.4760', 'uedin-nmt.4955']
    elif args.lang_pair == 'de-en':
        sys_list = ['C-3MA.4958', 'online-A.0', 'online-G.0', 'TALP-UPC.4830', 'KIT.4951', 'online-B.0', 'RWTH-nmt-ensemble.4920', 'uedin-nmt.4723', 'LIUM-NMT.4733', 'online-F.0', 'SYSTRAN.4846']
    elif args.lang_pair == 'fi-en':
        sys_list = ['apertium-unconstrained.4793', 'online-A.0', 'online-G.0', 'Hunter-MT.4925', 'online-B.0', 'TALP-UPC.4937']
    elif args.lang_pair == 'lv-en':
        sys_list = ['C-3MA.5067', 'online-A.0', 'tilde-c-nmt-smt-hybrid.5051', 'Hunter-MT.5092', 'online-B.0', 'tilde-nc-nmt-smt-hybrid.5050', 'jhu-pbmt.4980', 'PJATK.4740', 'uedin-nmt.5017']
    elif args.lang_pair == 'ru-en':
        sys_list = ['afrl-mitll-opennmt.4896', 'jhu-pbmt.4978', 'online-A.0', 'online-F.0', 'uedin-nmt.4890', 'afrl-mitll-syscomb.4905', 'NRC.4855', 'online-B.0', 'online-G.0']
    elif args.lang_pair == 'tr-en':
        sys_list = ['afrl-mitll-m2w-nr1.4901', 'JAIST.4859', 'LIUM-NMT.4888', 'online-B.0', 'PROMT-SMT.4737', 'afrl-mitll-syscomb.4902', 'jhu-pbmt.4972', 'online-A.0', 'online-G.0', 'uedin-nmt.4931']
    elif args.lang_pair == 'zh-en':
        sys_list = ['afrl-mitll-opennmt.5109', 'NRC.5172', 'online-G.0', 'SogouKnowing-nmt.5171', 'CASICT-cons.5144', 'online-A.0', 'Oregon-State-University-S.5173', 'uedin-nmt.5112', 'jhu-nmt.5151', 'online-B.0', 'PROMT-SMT.5125', 'UU-HNMT.5162', 'NMT-Model-Average-Multi-Cards.5099', 'online-F.0', 'ROCMT.5167', 'xmunmt.5160']
    elif args.lang_pair == 'en-ru':
        sys_list = ['afrl-mitll-backtrans.4907', 'online-B.0', 'online-H.0', 'jhu-pbmt.4986', 'online-F.0', 'PROMT-Rule-based.4736', 'online-A.0', 'online-G.0', 'uedin-nmt.4756']

    path_ref = '/clwork/shimanaka/Data/WMT17_MetricsTask/wmt17-metrics-task/wmt17-submitted-data/txt/references/newstest2017-{}-ref.{}'.format(src_lang + trg_lang, trg_lang)

    reference_list = [line.strip() for line in open(path_ref)]
    
    metric_score_path = '../wmt17-metrics-task-package/final-metric-scores/{}.seg.score'.format(args.metric)
    metric_score_dict = dict()
    for line in open(metric_score_path):
        lang, task, sys, sent_ID, score = line.strip().split('\t')[1:6]
        if lang != args.lang_pair or task != 'newstest2017':
            continue
        if sys not in metric_score_dict.keys():
            metric_score_dict[sys] = list()
        metric_score_dict[sys].append(score)

    sysout_list_dict = dict()
    for sys in sys_list:
        path_sys = '/clwork/shimanaka/Data/WMT17_MetricsTask/wmt17-metrics-task/wmt17-submitted-data/txt/system-outputs/newstest2017/{0}/newstest2017.{1}.{0}'.format(args.lang_pair, sys)
        sysout_list_dict[sys] = [line.strip() for line in open(path_sys)]

    path_score = 'newstest2017-segment-level-human/anon-proc-hits-seg-{}/analysis/ad-seg-scores.csv'.format(trg_lang)
    human_scores = list()
    metric_scores = list()
    for line in open(path_score):
        if line.split()[0] == src_lang and line.split()[1] == trg_lang:
            sent_ID = int(line.split()[5])
            sys = line.split()[6]
            if '+' in sys:
                sys = sys.split('+')[0]
            # if '+' in sys:
            #     sys = sys.split('+')
            #     print(sysout_list_dict[sys[0]][sent_ID - 1])
            #     print(sysout_list_dict[sys[1]][sent_ID - 1])
            score = line.split()[-1]
            human_scores.append(float(score))
            metric_scores.append(float(metric_score_dict[sys][sent_ID - 1]))
            # print('\t'.join([reference_list[sent_ID - 1], sysout_list_dict[sys][sent_ID - 1], score]))
        else:
            continue

    for score in metric_scores:
        print(score)
