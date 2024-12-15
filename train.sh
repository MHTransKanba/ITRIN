for i in 'twitter2015' 'twitter2017'
do
    echo ${i}
    python train.py --dataset ${i} \
    --data_dir ./data/Sentiment_Analysis/ \
    --VG_data_dir ./data/Image_Target_Matching/ \
    --imagefeat_dir ./data/twitter_images/ \
    --VG_imagefeat_dir ./data/twitter_images/ \
    --output_dir log/
done




