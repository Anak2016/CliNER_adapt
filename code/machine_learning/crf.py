######################################################################
#  CliCon - crf.py                                                   #
#                                                                    #
#  Willie Boag                                      wboag@cs.uml.edu #
#                                                                    #
#  Purpose: Implement CRF (using python-crfsuite)                    #
######################################################################


import sys
import os
import tempfile
import pycrfsuite


from tools import compute_performance_stats
from feature_extraction.read_config import enabled_modules

cliner_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
tmp_dir = os.path.join(cliner_dir, 'data', 'tmp')
def format_features(rows, labels=None):
    '''

    :param rows:
    :param labels:
    :return retVal = [[name_1 = featureVal1, name_2= featureVal2,...],....]
    '''
    retVal = []

    # For each line
    for i,line in enumerate(rows):
        # print(rows)
        '''
        [<4x12805 sparse matrix of type '<class 'numpy.float64'>'
            with 297 stored elements in Compressed Sparse Row format>, <8x12805 sparse matrix of type '<class 'numpy.float64'>'
        '''
        # print(line) # 1 line : 34 features
        # print(type(line)) #<class 'scipy.sparse.csr.csr_matrix'>
        '''
        (0, 59)          1.0
        (0, 237)         1.0
                ...
        (34, 10203)      1.0
        (34, 10455)      5.0
        '''
        # For each word in the line
        for j,features in enumerate(line):

            # print(type(features)) # <class 'scipy.sparse.csr.csr_matrix'>
            '''
            row = features_number
            (row , col)   val
            
            (0, 5671)     1.0
            (0, 5680)     1.0
            (0, 5695)     1.0
            '''
            # Nonzero dimensions
            inds  = features.nonzero()[1]
            # print(inds)
            '''
            [   22   243   308   351   352 ... ]
            '''
            values = []
            if labels:
                values.append( str(labels[i][j]) )

            # Value for each dimension
            for k in inds:
                values.append( '%d=%d' %  (k, features[0,k]))
            retVal.append("\t".join(values).strip())

        # Sentence boundary seperator
        retVal.append('')


    # print(type(retVal)) # list
    # print(retVal)
    '''
    ['0\t6=1\t231=1\t239=1\t334=1\t335=1\t340=1\t341=1\t34',..., ]
    '''
    # print(retVal[2])
    '''
     0       228=1   247=1   346=1   354=2   355=1   357=1   358=1   517=0   584=0   666=0   685=3   686=0   688=0   689=0   833=0   1081=0  1093=0  1270=0  1298=0  1321=0  1419=0  1423=0  14
     50=0    1471=0  1510=0  1663=0  1990=0  2220=0  2366=0
    '''

    # Sanity check
    '''
    global count
    if labels:
        out_f = 'a.txt' + str(count)
        start =  0 # 2
    else:
        out_f = 'b.txt' + str(count)
        start = 0
    count += 1
    with open(out_f, 'w') as f:
        for line in retVal:
            print >>f, line[start:]
    '''


    return retVal




def pycrf_instances(fi, labeled):
    '''
    ANAK
    fi is something like
    line in fi = ['label word1 word2 word3....']

    create xfeat and yfeat
    '''
    xseq = []
    yseq = []

    # Skip first element
    if labeled:
        begin = 1
    else:
        begin = 0

    for line in fi:
        line = line.strip('\n')
        if not line:
            # An empty line presents an end of a sequence.
            if labeled:
                yield xseq, tuple(yseq)

            else:
                yield xseq

            xseq = []
            yseq = []
            continue

        # Split the line with TAB characters.
        fields = line.split('\t')
        # Append the item to the item sequence.
        feats = fields[begin:]
        xseq.append(feats)
        # print(xseq)
        '''
        [['35=1', '243=1', '270=1', '351=6', '352=1',..],..] 
        '''
        # Append the label to the label sequence.
        if labeled:
            yseq.append(fields[0])


def train(X, Y, val_X=None, val_Y=None, test_X=None, test_Y=None):
    '''
    train()
    Train a Conditional Random Field for sequence tagging.
    
    @param X.     List of sparse-matrix sequences. Each sequence is one sentence.
    @param Y.     List of sequence tags. Each sequence is the sentence's per-token tags.
    @param val_X. More X data, but a heldout dev set.
    @param val_Y. More Y data, but a heldout dev set.
    @return A tuple of encoded parameter weights and hyperparameters for predicting.
    '''

    # Sanity Check detection: features & label
    #with open('a','w') as f:
    #    for xline,yline in zip(X,Y):
    #        for x,y in zip(xline,yline):
    #            print >>f, y, '\t', x.nonzero()[1][0]
    #        print >>f

    # Format features fot crfsuite
    feats = format_features(X,Y)
    # print(feats)# [... ,'' ,t12193=1\t12199=1', ...]

    # Create a Trainer object.
    trainer = pycrfsuite.Trainer(verbose=False)
    trainer.select(algorithm= 'pa') # algorithm:{‘lbfgs’, ‘l2sgd’, ‘ap’, ‘pa’, ‘arow’}

    # print(trainer.params()) # ['feature.minfreq', 'feature.possible_states', 'feature.possible_transitions', 'c1', 'c2', 'max_iterations', 'num_memories', 'epsilon', 'period', 'delta', 'linesearch', 'max_linesearch']
    # print(trainer.get_params())
    '''
    {'feature.minfreq': 0.0, 'feature.possible_states': False, 'feature.possible_transitions': False, 'type': 1, 'c': 1.0, 'error_sensitive': True, 'averaging': True, 'max_iterations': 100,
    'epsilon': 0.0}
    '''
    for xseq, yseq in pycrf_instances(feats, labeled=True):
        trainer.append(xseq, yseq)


    # Train the model
    os_handle,tmp_file = tempfile.mkstemp(dir=tmp_dir, suffix="crf_temp") #how is tmp_file created?
    #tmp_dir = cliner_dir/data/tmp #
    trainer.train(tmp_file)# save temporary file to C:\Users\Anak\PycharmProjects\CliNER\data\tmp\tmp02nmns01crf_temp

    # print(trainer.logparser.last_iteration)
    #{'num': 54, 'scores': {}, 'loss': 49.276289, 'feature_norm': 5.78118, 'error_norm': 0.041626, 'active_features': 16870, 'linesearch_trials': 1, 'linesearch_step': 1.0, 'time': 0.003}

    # Read the trained model into a string (so it can be pickled)
    model = ''
    with open(tmp_file, 'rb') as f:
        model = f.read()
    os.close(os_handle)

    # print(model) # b'x01\x00\x00\x00\xd1D\x00\x00\x01\x00\x00\x00\xd2D\x00\x00\x01\x00\x00\x00\xd3D\x00
    # print(X)
    '''
    [<2x12206 sparse matrix of type '<class 'numpy.float64'>'
        with 70 stored elements in Compressed Sparse Row format>, <9x12206 sparse matrix of type '<class 'numpy.float64'>'
    '''

    # Remove the temporary file
    os.remove(tmp_file)

    ######################################################################
    # information about fitting the model
    scores = {}

    # how well does the model fit the training data?
    train_pred = predict(model,     X) # ANAK
    train_stats = compute_performance_stats('train', train_pred, Y)
    scores['train'] = train_stats

    if val_X:
        val_pred  = predict(model, val_X)
        val_stats = compute_performance_stats('dev', val_pred, val_Y)
        scores['dev'] = val_stats

    if test_X:
        test_pred  = predict(model, test_X)
        test_stats = compute_performance_stats('test', test_pred, test_Y)
        scores['test'] = test_stats

    # keep track of which external modules were used for building this model!
    scores['hyperparams'] = {}
    enabled_mods = enabled_modules()
    for module,enabled in enabled_mods.items():
        e = bool(enabled)
        scores['hyperparams'][module] = e

    # print(len(scores)) # 3
    return model, scores


def predict(clf, X):

    # Format features fot crfsuite
    feats = format_features(X)
    # print(feats)
    '''
    ['0\t6=1\t231=1\t239=1\t334=1\t335=1\t340=1\t341=1\t34',..., ]
    '''

    # Dump the model into a temp file
    os_handle,tmp_file = tempfile.mkstemp(dir=tmp_dir, suffix="crf_temp")
    with open(tmp_file, 'wb') as f:
        # clf_byte = bytearray(clf, 'utf8')
        clf_byte = bytearray(clf)
        f.write(clf_byte)

    # Create the Tagger object
    tagger = pycrfsuite.Tagger()
    tagger.open(tmp_file)


    # Remove the temp file
    os.close(os_handle)
    os.remove(tmp_file)

    # probability
    # print(tagger.labels())
    # dump
    # exit()


    # Tag the sequence
    retVal = []
    Y = []
    for xseq in pycrf_instances(feats, labeled=False):
        # print(tagger.probability(['66']))
        # exit()

        # print(xseq) # represent a sentence? if not what?
        # exit()
        '''
        [['66=1', '240=1', '286=1', '347=10', '348=1', '350=1', '351=1', '364=0', '572=0', '581=0', '672=0', '673=0']]
        '''
        #input to a tagger.tag must be a sentence, but number of elements are not equal.
        yseq = [ int(n) for n in tagger.tag(xseq) ]
        # tagger.set(xseq)
        # print(tagger.probability(xseq))
        # print(tagger.marginal(xseq[0],0))
        # print(len(xseq[0]))
        # exit()

        # yseq = [ n for n in tagger.tag(xseq) ] ['0', '0', '0', '0', '1', '4', '4', '4', 0']
        # print(yseq)
        # exit()
        '''
        1 INDEX = 1 SENTENCE??
        [0, 0, 0, 0, 1, 4, 4, 4, 0] 
        '''
        retVal += list(yseq)
        Y.append(list(yseq))

    # print(Y) # [[0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 4, 4, 4, 0], ...]

    # Sanity Check detection: feature & label predictions
    #with open('a','w') as f:
    #    for x,y in zip(xseq,Y):
    #        x = x[0]
    #        print >>f, y, '\t', x[:-2]
    #    print >>f


    return Y
