for i in 'twitter2017' # 'twitter2017'
do
    python test.py --dataset ${i} \
    --data_dir ./data/ \
    --VG_data_dir ./data/Image_Target_Matching/ \
    --imagefeat_dir ./data/twitter_images/ \
    --VG_imagefeat_dir ./data/twitter_images/ \
    --output_dir ./log/ \
    --model_file ./log/${i}/pytorch_model.bin
done

