#! /bin/bash

data=PATH_TO_DATA #`pwd`
bin=PATH_TO_CODE #`pwd`
gpu=0

mkdir test
cd test
python ${bin}/code/thumt/scripts/checkpoint_averaging.py --path ${bin}/train --checkpoints 5 --output avg
python ${bin}/translator.py --models transformer --input $data/dev.32k.en \
        --output dev.trans \
        --checkpoints $bin/test/avg \
        --vocabulary $data/vocab.32k.en.txt $data/vocab.32k.de.txt \
        --parameters=device_list=[$gpu]
sed "s/@@ //g" < dev.trans > dev.trans.bpe
$data/multi-bleu.perl $data/dev.de <  dev.trans.bpe >  dev.trans.bpe.nmt

python $bin/translator.py --models transformer --input $data/newstest2014.32k.en \
        --output newstest2014.trans \
        --checkpoints $bin/test/avg \
        --vocabulary $data/vocab.32k.en.txt $data/vocab.32k.de.txt \
        --parameters=device_list=[$gpu]
sed "s/@@ //g" < newstest2014.trans > newstest2014.trans.bpe
$data/multi-bleu.perl $data/newstest2014.de <  newstest2014.trans.bpe >  newstest2014.trans.bpe.nmt


