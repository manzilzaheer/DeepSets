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
#LOAD_PATH=models/model_1k.pth
LOAD_PATH=models/model_3k.pth
#LOAD_PATH=models/model_5k.pth
#LOAD_PATH=models/models-13-Apr-2018-06:33:47/best_model.t7
#LOAD_PATH=models/models-14-Apr-2018-08:28:44/best_model.t7

python -u test.py -inputData ${INPUT_DATA} -loadPath ${LOAD_PATH} -dataset lda -evalSize 5

