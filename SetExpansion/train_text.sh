# bash script for running code

# check if GPU id passed in argument
GPU_ID=$1
re='^[0-3]+$'
if ! [[ $1 =~ $re ]] ; then
   GPU_ID=0
fi

export CUDA_VISIBLE_DEVICES=${GPU_ID}
echo 'Using GPU: '${GPU_ID}


# LDA path
ROOT='./data/'
#INPUT_DATA=${ROOT}'lda/lda_top_words_split_allwords_1k.json'
INPUT_DATA=${ROOT}'lda/lda_top_words_split_allwords_3k.json'
#INPUT_DATA=${ROOT}'lda/lda_top_words_split_allwords_5k.json'

# lda
MODEL='w2v_sum'
MARGIN=0.1
# training
python -u train.py -inputData ${INPUT_DATA} -margin ${MARGIN} -modelName ${MODEL} -dataset lda -learningRate 0.0001 -evalSize 5 -evalPerEpoch 10 -numEpochs 10000 -embedSize 50
