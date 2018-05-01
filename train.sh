#! /bin/bash

data=PATH_TO_DATA #`pwd`
python trainer.py --input $data/train.32k.en.shuf $data/train.32k.de.shuf \
        --model transformer --output train \
        --vocabulary $data/vocab.32k.en.txt $data/vocab.32k.de.txt \
        --validation $data/dev.32k.en \
        --references $data/dev.32k.de \
        --parameters=batch_size=3125,device_list=[0],eval_steps=5000,train_steps=100000,save_checkpoint_steps=1500,shared_embedding_and_softmax_weights=true,shared_source_target_embedding=false,update_cycle=8
