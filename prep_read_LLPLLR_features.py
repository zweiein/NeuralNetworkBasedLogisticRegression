# -*- coding: utf-8 -*-

import codecs, sys


def GetPLFeats(kaldi_trunk_path, abs_pl_feats_path):
#return phone_LLPR_matrix
#        #phone_LLPR_matrix資料架構長這樣： [<syllable>, <T or F>, <list_ofLPP_feature>, <list_of_LPR_feature>]
#        #總長度436，LPP跟LPR特徵皆有217維(217*2+2=436)
#        #['in1', 'T', '-29.3189127821', '-37.9716765342', '-40.4703022908', ...]
#        #['l', 'T', '-21.1822492549', '-16.2802555299', '-15.4576691132', ...]
#        # ...
#        #'h', 'T', '-14.2529502402', '-17.0329784402', '-12.0516192402', ...]
    dev_mode = True
    if dev_mode :
        #base_path = '/Users/slp/Documents/'
        kaldi_trunk_path = '/usr/local/kaldi-trunk/'
        pl_feats_path = '/share/homes/yang/tupntnu_DS/sR/exp_rnn_lstm/delta2_and_smbr/PLFeats_DS_dev/phone_level_corpus.txt'
        #exp_json_path = '/Users/slp/Documents/TopNTNUCorpus/DataPrepare/output_json/plf_dev.json'
    #end dev_mode
    else : #online mode
        kaldi_trunk_path = kaldi_trunk_path
        pl_feats_path = abs_pl_feats_path
        #exp_json_path = sys.argv[3]
    #end online mode

    phone_LLPR_matrix = []

    with open(pl_feats_path) as read_file:
        this_line = read_file.readline()
        while this_line :
            #值得注意的是，同一個phone，其LLP與LLR也不見得相同！
            #每一行的格式：<phoneme> <T or F> <LLP features> <LLR features>
            #     a3 T -13.4104575982 -10.2342599648 -5.53997163151 -11.4920939648 ... -5.585625446
            #     sh T -19.0590498419 -17.8852997863 -19.3361981197 -19.1589873419 -16.9709558974 ...
            #     ing4 T -23.4679870636 -22.4070665874 -27.7336907145 ...
            #     empt25 T -21.0460634168 -22.8201075242 -20.9945517131 -25.7833834705 -17.213965639 -24.5143787871 ...
            this_line_sp = this_line.split(' ')
            this_line_sp[-1] = this_line_sp[-1].replace('\n', '')  #spit後，最後一個元素內含換行字元，要拿掉
            phone_LLPR_matrix.append(this_line_sp)
            this_line = read_file.readline()
        #end for every line
    #end with read file

    return phone_LLPR_matrix
#end GetPLFeats()



if __name__ == '__main__':
    dev_mode = True
    # check the arguments
    if len(sys.argv) != 4 and not dev_mode :
        print '  Usage: python prep_read_LLPLLR_features.py <kaldi_trunk_path> <pl_feats_path> <exp_json_path>\n'
        print '  ex: python prep_read_LLPLLR_features.py  \n'
        print '  ex:            /home/yang/kaldi-trunk/ \n'
        print '                 /home/yang/kaldi-trunk/egs/tupntnu_MS/s1/exp/PLFeats_DS_dev/phone_level_corpus.txt\n'
        print '                  /home/yang/kaldi-trunk/egs/output_json_path/plf_dev.json\n'
        exit(1)
    #end if arg incorrect  /Users/slp/Documents/kaldi-trunk/egs/tupntnu_MS/sR/exp_rnn_lstm/delta2_and_smbr/PLFeats_DS_dev
    else :
        if dev_mode :
            base_path = '/Users/slp/Documents/'
            kaldi_trunk_path = base_path + 'kaldi-trunk/'
            pl_feats_path = kaldi_trunk_path + 'egs/tupntnu_MS/sR/exp_rnn_lstm/delta2_and_smbr/PLFeats_DS_dev/phone_level_corpus.txt'
            exp_json_path = '/Users/slp/Documents/TopNTNUCorpus/DataPrepare/output_json/plf_dev.json'
        #end dev_mode
        else : #online mode
            kaldi_trunk_path = sys.argv[1]
            pl_feats_path = sys.argv[2]
            exp_json_path = sys.argv[3]
        #end online mode

        phone_LLPR_matrix = []

        with open(pl_feats_path) as read_file:
            this_line = read_file.readline()
            while this_line :
                #值得注意的是，同一個phone，其LLP與LLR也不見得相同！
                #格式：<phoneme> <T or F> <LLP features> <LLR features>
                #     a3 T -13.4104575982 -10.2342599648 -5.53997163151 -11.4920939648 ... -5.585625446
                #     sh T -19.0590498419 -17.8852997863 -19.3361981197 -19.1589873419 -16.9709558974 ...
                #     ing4 T -23.4679870636 -22.4070665874 -27.7336907145 ...
                #     empt25 T -21.0460634168 -22.8201075242 -20.9945517131 -25.7833834705 -17.213965639 -24.5143787871 ...
                this_line_sp = this_line.split(' ')
                this_line_sp[-1] = this_line_sp[-1].replace('\n', '')  #spit後，最後一個元素內含換行字元，要拿掉
                phone_LLPR_matrix.append(this_line_sp)

                this_line = read_file.readline()
            #end for every line
        #end with read file

        #phone_LLPR_matrix資料架構長這樣： [<syllable>, <T or F>, <list_ofLPP_feature>, <list_of_LPR_feature>]
        #總長度436，LPP跟LPR特徵皆有217維(217*2+2=436)
        #['in1', 'T', '-29.3189127821', '-37.9716765342', '-40.4703022908', ...]
        #['l', 'T', '-21.1822492549', '-16.2802555299', '-15.4576691132', ...]
        # ...
        #'h', 'T', '-14.2529502402', '-17.0329784402', '-12.0516192402', ...]

        print phone_LLPR_matrix[0]
        print '====='
        print len(phone_LLPR_matrix[1])
        print (len(phone_LLPR_matrix[1]) -2)/2
        print phone_LLPR_matrix[1]
        print '====='
        print len(phone_LLPR_matrix[15])
        print (len(phone_LLPR_matrix[15]) -2)/2
        print phone_LLPR_matrix[15]
        print '===='
        print len(phone_LLPR_matrix[50])
        print (len(phone_LLPR_matrix[50]) -2)/2
        print phone_LLPR_matrix[50]
#end if __name__ == '__main__':