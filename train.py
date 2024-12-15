import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置采用的GPU序号
import logging
import argparse
import random
import datetime
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from itertools import cycle
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics import precision_recall_fscore_support
from torch import optim
from torch.nn import CrossEntropyLoss

from DataProcessor import *
from model import Coarse2Fine
from boxes_utils import *
from optimization import BertAdam


def macro_f1(y_true, y_pred):
    preds = np.argmax(y_pred, axis=-1)
    true = y_true
    p_macro, r_macro, f_macro, support_macro \
        = precision_recall_fscore_support(true, preds, average='macro')
    # f_macro = 2*p_macro*r_macro/(p_macro+r_macro)
    return p_macro, r_macro, f_macro


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


def post_dataloader(batch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokens, input_ids, input_mask, sentiment_label, \
        img_id, img_shape, relation_label, GT_boxes, roi_boxes, img_feat, spatial_feat, box_labels = batch

    input_ids = list(map(list, zip(*input_ids)))
    input_mask = list(map(list, zip(*input_mask)))
    img_shape = list(map(list, zip(*img_shape)))

    input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
    input_mask = torch.tensor(input_mask, dtype=torch.long).to(device)
    img_shape = torch.tensor(img_shape, dtype=torch.float).to(device)
    sentiment_label = sentiment_label.to(device).long()
    relation_label = relation_label.to(device).long()
    GT_boxes = GT_boxes.to(device).float()
    roi_boxes = roi_boxes.to(device).float()
    img_feat = img_feat.to(device).float()
    spatial_feat = spatial_feat.to(device).float()
    box_labels = box_labels.to(device).float()

    return tokens, input_ids, input_mask, sentiment_label, \
        img_id, img_shape, relation_label, GT_boxes, roi_boxes, img_feat, spatial_feat, box_labels


def main():
    start_time = datetime.datetime.now().strftime('%m-%d-%Y-%H-%M-%S_')
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--dataset",
                        default='twitter2015',
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--data_dir",
                        default='./data/',
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files for the task.")
    parser.add_argument("--VG_data_dir",
                        default='./data/Image_Target_Matching',
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files for the task.")
    parser.add_argument("--imagefeat_dir",
                        default='./data/twitter_images/',  # default ='./data/twitter_images/',
                        type=str,
                        required=True,
                        )
    parser.add_argument("--VG_imagefeat_dir",
                        default='./data/twitter_images/',  # default ='./data/twitter_images/',
                        type=str,
                        required=True,
                        )
    parser.add_argument("--output_dir",
                        default="./log/",
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--save",
                        default=True,
                        action='store_true',
                        help="Whether to save model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--SA_learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--VG_learning_rate",
                        default=1e-6,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--ranking_loss_ratio",
                        default=0.5,
                        type=float)
    parser.add_argument("--pred_loss_ratio",
                        default=1.,
                        type=float)
    parser.add_argument("--num_train_epochs",
                        default=12.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%  of training.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,  # 42
                        help="random seed for initialization")
    parser.add_argument('--roi_num',
                        default=100,
                        type=int)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.data_dir = args.data_dir + str(args.dataset).lower() + '/%s.pkl'
    args.imagefeat_dir = args.imagefeat_dir + str(args.dataset).lower()
    args.VG_data_dir = args.VG_data_dir + '/%s.pkl'
    args.VG_imagefeat_dir = args.VG_imagefeat_dir + 'twitter2017'  # image-target-matching data from twitter2017

    args.output_dir = args.output_dir + str(args.dataset) + "/"

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    output_logger_file = os.path.join(args.output_dir, 'log.txt')

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        filename=output_logger_file)
    logger = logging.getLogger(__name__)

    logger.info("dataset:{}   num_train_epochs:{}".format(args.dataset, args.num_train_epochs))
    logger.info("SA_learning_rate:{}  warmup_proportion:{}".format(args.SA_learning_rate, args.warmup_proportion))
    logger.info("VG_learning_rate:{}   warmup_proportion:{}".format(args.VG_learning_rate, args.warmup_proportion))
    logger.info(args)

    # 通过设置相同的随机数种子，可以确保实验具有可重复性、可控性，并且更容易进行结果比较和理解模型行为。
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    local_model_path = "./roberta"
    tokenizer = RobertaTokenizer.from_pretrained(local_model_path)

    # VG需把细粒度方面词目标框对齐删除，SA需要全删除
    train_dataset_SA = MyDataset(args.data_dir % str('train'), args.imagefeat_dir, tokenizer,
                                 max_seq_len=args.max_seq_length, num_roi_boxes=100)
    train_dataset_VG = MyDataset(args.VG_data_dir % str('VG_train'), args.VG_imagefeat_dir, tokenizer,
                                 max_seq_len=args.max_seq_length, num_roi_boxes=100)
    train_dataloader_SA = Data.DataLoader(dataset=train_dataset_SA, shuffle=True, batch_size=args.train_batch_size,
                                          num_workers=0)
    train_dataloader_VG = Data.DataLoader(dataset=train_dataset_VG, shuffle=True, batch_size=args.train_batch_size,
                                          num_workers=0)

    eval_dataset_SA = MyDataset(args.data_dir % str('dev'), args.imagefeat_dir, tokenizer,
                                max_seq_len=args.max_seq_length, num_roi_boxes=100)
    eval_dataset_VG = MyDataset(args.VG_data_dir % str('VG_dev'), args.VG_imagefeat_dir, tokenizer,
                                max_seq_len=args.max_seq_length, num_roi_boxes=100)
    eval_dataloader_SA = Data.DataLoader(dataset=eval_dataset_SA, shuffle=False, batch_size=args.eval_batch_size,
                                         num_workers=0)
    eval_dataloader_VG = Data.DataLoader(dataset=eval_dataset_VG, shuffle=False, batch_size=args.eval_batch_size,
                                         num_workers=0)

    test_dataset = MyDataset(args.data_dir % str('test'), args.imagefeat_dir, tokenizer,
                             max_seq_len=args.max_seq_length, num_roi_boxes=100)
    test_dataloader = Data.DataLoader(dataset=test_dataset, shuffle=False, batch_size=args.eval_batch_size,
                                      num_workers=0)

    # 首先，将train_number（训练数据集中样本的最大数量）除以args.train_batch_size（训练批次大小）得到每个轮次中的批次数量。
    # 然后，将上一步得到的批次数量乘以args.num_train_epochs（训练轮次数），得到总的训练步数
    train_number = max(train_dataset_SA.number, train_dataset_VG.number)
    num_train_steps = int(train_number / args.train_batch_size * args.num_train_epochs)

    model = Coarse2Fine(roberta_name=local_model_path)
    model.to(device)

    # new_state_dict=model.state_dict()
    # logger.info(new_state_dict)

    # Prepare optimizer
    # optimizer BertAdam

    # 在深度学习中，权重衰减（weight decay）是一种正则化技术，通过在优化器的损失函数中添加一个正则项来防止模型过拟合。正则项一般表示为权重的平方和，并乘以一个调整参数
    # 给定的代码中，通过将模型的参数分成两组，一组是需要进行权重衰减的参数（例如卷积层和全连接层的权重），另一组是不需要进行权重衰减的参数（例如偏置项和 LayerNormalization 层的参数）。
    # 这样的设置允许在训练过程中对这两组参数应用不同的权重衰减策略，以提高模型对数据的拟合效果，并减少过拟合的可能性。
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # warmup 表示学习率的预热步骤。在训练初期，逐渐增加学习率有助于模型更稳定地收敛到最优解。args.warmup_proportion 是从命令行参数中获取的预热比例。
    # t_total 表示总的训练步数。这是一个重要的参数，它决定了学习率的变化方式。num_train_steps 表示从数据集中训练一个周期所需的步数。
    # 该模型俩个优化器使用的是一样的参数即模型全部参数，为了满足实验思路要求，更新前后顺序有区别，先粗细粒度匹配带来的损失，然后是情感标签预测的损失
    optimizer_VG = BertAdam(optimizer_grouped_parameters,
                            lr=args.VG_learning_rate,
                            warmup=args.warmup_proportion,
                            t_total=num_train_steps)

    optimizer_SA = BertAdam(optimizer_grouped_parameters,
                            lr=args.SA_learning_rate,
                            warmup=args.warmup_proportion,
                            t_total=num_train_steps)

    VG_global_step = 0
    SA_global_step = 0
    nb_tr_steps = 0
    max_senti_acc = 0.0
    best_epoch = -1

    # 这是一个循环，遍历每个训练周期。trange 函数用于在循环中显示进度条，desc="Epoch" 是进度条的描述文本。
    logger.info("*************** Running training ***************")
    for train_idx in trange(int(args.num_train_epochs), desc="Epoch"):

        logger.info("************************************************** Epoch: " + str(
            train_idx) + " *************************************************************")
        logger.info("  Num examples = %d", train_number)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        ### train
        model.train()
        pred_l = 0
        ranking_l, senti_l = 0, 0
        # 利用 zip 函数将两个数据迭代器 train_dataloader_VG 和 train_dataloader_SA 进行配对，同时使用 cycle 使得 train_dataloader_VG 在每个训练周期中都能被无限循环使用。
        # 这样，就能够在每个步骤中获取到配对的数据，实现对两个数据集同时迭代的效果。
        # tqdm 是一个 Python 的进度条库，可以在循环迭代中显示进度条，方便用户了解任务的进度。在代码中，tqdm 用于显示训练步骤的进度。
        for step, data in enumerate(tqdm(zip(cycle(train_dataloader_VG), train_dataloader_SA), desc="Iteration")):
            batch_VG, batch_SA = data

            #### VG
            tokens, input_ids, input_mask, sentiment_label, \
                img_id, img_shape, relation_label, GT_boxes, roi_boxes, img_feat, spatial_feat, box_labels = post_dataloader(
                batch_VG)

            senti_pred, pred_loss, pred_score = model(img_id=img_id,
                                                      input_ids=input_ids,
                                                      input_mask=input_mask,
                                                      img_feat=img_feat,
                                                      relation_label=relation_label,
                                                      pred_loss_ratio=args.pred_loss_ratio,
                                                      )
            loss_VG = pred_loss
            loss_VG.backward()

            # 预热步骤的长度通常由预热比例参数决定，即训练总步数的一小部分。在预热阶段，学习率从一个很小的值逐渐增加到初始学习率。
            # 预热步骤的主要目的是在训练初期避免模型收敛到局部最优解，从而更好地探索参数空间。
            lr_this_step = args.VG_learning_rate * warmup_linear(VG_global_step / num_train_steps,
                                                                 args.warmup_proportion)
            for param_group in optimizer_VG.param_groups:
                param_group['lr'] = lr_this_step
            optimizer_VG.step()
            optimizer_VG.zero_grad()
            VG_global_step += 1

            pred_l += pred_loss.item()

            #### SA
            SA_tokens, SA_input_ids, SA_input_mask, SA_sentiment_label, \
                SA_img_id, SA_img_shape, SA_relation_label, SA_GT_boxes, SA_roi_boxes, SA_img_feat, SA_spatial_feat, SA_box_labels = post_dataloader(
                batch_SA)

            senti_pred, pred_loss, pred_score = model(img_id=SA_img_id,
                                                      input_ids=SA_input_ids,
                                                      input_mask=SA_input_mask,
                                                      img_feat=SA_img_feat,
                                                      relation_label=None,
                                                      )

            senti_loss_fct = CrossEntropyLoss()
            sentiment_loss = senti_loss_fct(senti_pred.view(-1, 3), SA_sentiment_label.view(-1))
            loss_SA = sentiment_loss
            loss_SA.backward()

            senti_l += sentiment_loss.item()

            lr_this_step = args.SA_learning_rate * warmup_linear(SA_global_step / num_train_steps,
                                                                 args.warmup_proportion)
            for param_group in optimizer_SA.param_groups:
                param_group['lr'] = lr_this_step
            optimizer_SA.step()
            optimizer_SA.zero_grad()
            SA_global_step += 1

            nb_tr_steps += 1

        pred_l = pred_l / nb_tr_steps
        senti_l = senti_l / nb_tr_steps

        logger.info("pred_loss:%s", pred_l)
        logger.info("sentiment_loss:%s", senti_l)

        ### dev
        model.eval()
        logger.info("***** Running evaluation on Dev Set*****")
        logger.info("  SA Num examples = %d", eval_dataset_SA.number)  # len(eval_examples)
        logger.info("  Batch size = %d", args.eval_batch_size)

        nb_eval_examples = 0
        SA_nb_eval_examples = 0
        senti_acc, rel_acc = 0, 0
        senti_precision, senti_recall, senti_F_score = 0, 0, 0
        senti_true_label_list = []
        senti_pred_label_list = []
        num_right_vg = 0
        num_valid = 0

        #### VG
        for s, batch_VG in enumerate(tqdm(eval_dataloader_VG, desc="Evaluating_VG")):
            tokens, input_ids, input_mask, sentiment_label, \
                img_id, img_shape, relation_label, GT_boxes, roi_boxes, img_feat, spatial_feat, box_labels = post_dataloader(
                batch_VG)

            with torch.no_grad():
                senti_pred, pred_loss, pred_score = model(img_id=img_id,
                                                          input_ids=input_ids,
                                                          input_mask=input_mask,
                                                          img_feat=img_feat,
                                                          relation_label=relation_label,
                                                          )
            current_batch_size = input_ids.size()[0]

            # -----evaluate
            ##### coarse-grained
            pred_score = pred_score.detach().cpu().numpy()  # [N*n, 100]                 #.reshape(current_batch_size,args.max_GT_boxes,-1)  # [N*n, 100]->[N, n, 100]
            relation_pred = np.argmax(pred_score, axis=1)
            tmp_rel_accuracy = np.sum(relation_pred == relation_label.cpu().numpy())
            rel_acc += tmp_rel_accuracy

            # roi_boxes=roi_boxes.detach().cpu().numpy()  #[N, 100, 4]
            # GT_boxes=GT_boxes.detach().cpu()  #[N, n,4]
            # attn_map=attn_map.detach().cpu().numpy()

            ##### fine-grained
            # for i in range(current_batch_size): #N
            #     if relation_label[i]!=0:
            #         num_valid+=1
            #
            #         ious=(torchvision.ops.box_iou(GT_boxes[i,0:1,:],torch.tensor(roi_boxes[i]))).numpy() #[1,4],[100,4]->[1,100] #如果GT是0，iou为0
            #         sorted_index=np.argsort(-attn_map[i])[0]
            #         pred_ids=sorted_index[:1]  #top K=1
            #         topk_max_iou=ious[0][pred_ids]
            #         pred_iou=topk_max_iou.max()
            #
            #         if pred_iou>=0.5:
            #             num_right_vg+=1

            nb_eval_examples += current_batch_size

        rel_acc = rel_acc / nb_eval_examples

        #### SA
        for batch_SA in tqdm(eval_dataloader_SA, desc="Evaluating_SA"):
            SA_tokens, SA_input_ids, SA_input_mask, SA_sentiment_label, \
                SA_img_id, SA_img_shape, SA_relation_label, SA_GT_boxes, SA_roi_boxes, SA_img_feat, SA_spatial_feat, SA_box_labels = post_dataloader(
                batch_SA)

            with torch.no_grad():
                SA_senti_pred, SA_pred_loss, SA_pred_score = model(
                    img_id=SA_img_id,
                    input_ids=SA_input_ids,
                    input_mask=SA_input_mask,
                    img_feat=SA_img_feat,
                    relation_label=None,
                )

            SA_sentiment_label = SA_sentiment_label.cpu().numpy()
            SA_senti_pred = SA_senti_pred.cpu().numpy()
            senti_true_label_list.append(SA_sentiment_label)
            senti_pred_label_list.append(SA_senti_pred)
            tmp_senti_accuracy = accuracy(SA_senti_pred, SA_sentiment_label)
            senti_acc += tmp_senti_accuracy

            current_batch_size = SA_input_ids.size()[0]
            SA_nb_eval_examples += current_batch_size

        senti_acc = senti_acc / SA_nb_eval_examples

        senti_true_label = np.concatenate(senti_true_label_list)
        senti_pred_outputs = np.concatenate(senti_pred_label_list)
        senti_precision, senti_recall, senti_F_score = macro_f1(senti_true_label, senti_pred_outputs)

        result = {'nb_eval_examples': nb_eval_examples,
                  'num_valid': num_valid,
                  'Dev_rel_acc': rel_acc,
                  'Dev_senti_acc': senti_acc,
                  'Dev_senti_precision': senti_precision,
                  'Dev_senti_recall': senti_recall,
                  'Dev_senti_F_score': senti_F_score,
                  }
        logger.info("***** Dev Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

        ### test
        model.eval()
        logger.info("***** Running evaluation on Test Set *****")
        logger.info("  Num examples = %d", test_dataset.number)
        logger.info("  Batch size = %d", args.eval_batch_size)

        nb_test_examples = 0
        test_senti_acc = 0
        test_senti_true_label_list = []
        test_senti_pred_label_list = []
        for batch in tqdm(test_dataloader, desc="Testing"):
            tokens, input_ids, input_mask, sentiment_label, \
                img_id, img_shape, relation_label, GT_boxes, roi_boxes, img_feat, spatial_feat, box_labels = post_dataloader(
                batch)

            with torch.no_grad():
                senti_pred, pred_loss, pred_score = model(
                    img_id=img_id,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    img_feat=img_feat,
                    relation_label=None,
                )

            sentiment_label = sentiment_label.cpu().numpy()
            senti_pred = senti_pred.cpu().numpy()
            test_senti_true_label_list.append(sentiment_label)
            test_senti_pred_label_list.append(senti_pred)
            tmp_senti_accuracy = accuracy(senti_pred, sentiment_label)
            test_senti_acc += tmp_senti_accuracy

            current_batch_size = input_ids.size()[0]
            nb_test_examples += current_batch_size

        test_senti_acc = test_senti_acc / nb_test_examples

        senti_true_label = np.concatenate(test_senti_true_label_list)
        senti_pred_outputs = np.concatenate(test_senti_pred_label_list)
        test_senti_precision, test_senti_recall, test_senti_F_score = macro_f1(senti_true_label, senti_pred_outputs)

        result = {
            'Test_senti_acc': test_senti_acc,
            'Test_senti_precision': test_senti_precision,
            'Test_senti_recall': test_senti_recall,
            'Test_senti_F_score': test_senti_F_score}
        logger.info("***** Test Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

        # save model
        if senti_acc >= max_senti_acc:
            # Save a trained model
            if args.save:
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(), output_model_file)
            max_senti_acc = senti_acc
            corresponding_test_acc = test_senti_acc
            best_epoch = train_idx

    logger.info("max_dev_senti_acc: %s ", max_senti_acc)
    logger.info("corresponding_test_sentiment_acc: %s ", corresponding_test_acc)
    logger.info("corresponding_test_sentiment_precision: %s ", test_senti_precision)
    logger.info("corresponding_test_sentiment_recall: %s ", test_senti_recall)
    logger.info("corresponding_test_sentiment_F_score: %s ", test_senti_F_score)
    logger.info("best_epoch: %d", best_epoch)


if __name__ == "__main__":
    main()

# --dataset twitter2015 --data_dir ./data/Sentiment_Analysis/ --VG_data_dir ./data/Image_Target_Matching/ --imagefeat_dir ./data/twitter_images/ --VG_imagefeat_dir ./data/twitter_images/ --output_dir log/