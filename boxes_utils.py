import torch
import torchvision
import numpy as np
import os
import os.path as osp
def get_batch_visual_label(rois_boxess,GT_boxess):
    box_labels=[]
    target_labels=[]
    for i in range(len(rois_boxess)):
        box_label, target_label=get_visual_label(rois_boxess[i],GT_boxess[i])
        box_labels.append(box_label)
        target_labels.append(target_label)
    return torch.tensor(box_labels),torch.tensor(target_labels)


def get_visual_label(rois_boxes,GT_boxes):
    #rois_boxes:[100,4]
    #GT_boxes:[n,4]
    num_GT_boxes=len(GT_boxes)   #n
    num_roi_boxes=len(rois_boxes) #100
    
    box_label=np.zeros((num_GT_boxes,num_roi_boxes))  #[1,100]  #if IoU >= 0.5, =IoU_i/sum(); =0
    target_label=np.zeros((num_GT_boxes,num_roi_boxes,4)) #[1,100] #if IoU >= 0.5,= GT偏移量    ; =0
    box_pos =np.zeros((num_GT_boxes,num_roi_boxes))
    tensor_roi=torch.from_numpy(rois_boxes)
    tensor_GT=torch.from_numpy(GT_boxes)
    
    IoUs=(torchvision.ops.box_iou(tensor_GT,tensor_roi)).numpy() #[n,4],[100,4]  ->[n,100]
    #print("np.shape(IoUs): "+str(np.shape(IoUs)))
    
    gt_assignment = IoUs.argmax(axis=0)       
    gt_target_boxes = GT_boxes[gt_assignment, :4]  #[100] 各roi对应的“真”值

    for i,ious in enumerate(IoUs): # 多个GT_boxes
        if ious.max()>=0.5:
            pos_idx=np.where(ious>=0.5) 
            box_label[i,pos_idx]=ious[pos_idx]
            box_label[i]= box_label[i] / float(box_label[i].sum())

            box_pos[i,pos_idx]=1
            #gt_target_boxes[pos_idx] = GT_boxes[i,pos_idx, :4]  

        bbox_target_data = compute_targets( rois_boxes, gt_target_boxes, box_pos[i])
        target_label[i]=  get_query_bbox_regression_labels(bbox_target_data)
    
    return box_label,target_label


def get_attention_label(rois_boxes,GT_boxes):
    #rois_boxes:[n,100,4]
    #GT_boxes:[n,4]

    num_GT_boxes=len(GT_boxes)
    num_roi_boxes=len(rois_boxes) #100
    
    box_label=np.zeros((1,num_roi_boxes))  #[1,100]  #if IoU >= 0.5, =IoU_i/sum(); =0
    temp_i=np.zeros((1,num_roi_boxes)) 
   
    tensor_roi=torch.from_numpy(rois_boxes)
    tensor_GT=torch.from_numpy(GT_boxes)
    
    IoUs=(torchvision.ops.box_iou(tensor_GT,tensor_roi)).numpy() #[n,1,,4],[n,100,4]  ->[n,100]
    #print("np.shape(IoUs): "+str(np.shape(IoUs)))

    for i,ious in enumerate(IoUs): #多个GT_boxes
        if ious.max()>=0.5:
            pos_idx=np.where(ious>=0.5) 
            temp_i[0,pos_idx]=ious[pos_idx]
            temp_i[0]= temp_i[0] / float(temp_i[0].sum())
        # else:
        #     temp_i[0]=np.full((1,num_roi_boxes),0.01)

        box_label[0]+=temp_i[0]

    box_label[0]=box_label[0]/num_GT_boxes

    
    return box_label
# num_GT_boxes=len(GT_boxes) 和 num_roi_boxes=len(rois_boxes)：
#
# 这两行代码分别获取了真实边界框的数量和感兴趣区域的数量。
# box_label=np.zeros((1,num_roi_boxes)) 和 temp_i=np.zeros((1,num_roi_boxes))：
#
# 这两行代码创建了两个形状为 (1, num_roi_boxes) 的 NumPy 数组，用于存储最终的关注度标签和临时计算用的数组。
# tensor_roi=torch.from_numpy(rois_boxes) 和 tensor_GT=torch.from_numpy(GT_boxes)：
#
# 这两行代码将输入的感兴趣区域和真实边界框转换为 PyTorch 的 Tensor 格式，以便后续计算。
# IoUs=(torchvision.ops.box_iou(tensor_GT,tensor_roi)).numpy()：
#
# 这行代码使用 PyTorch 提供的 torchvision.ops.box_iou 函数计算了每个真实边界框与每个感兴趣区域之间的 IoU（Intersection over Union）。最后将结果转换为 NumPy 数组。
# for i, ious in enumerate(IoUs):：
#
# 这个循环遍历每个真实边界框的 IoU 值。i 表示当前真实边界框的索引，ious 是该边界框与所有感兴趣区域的 IoU 值。
# if ious.max() >= 0.5:：
#
# 这个条件判断语句检查当前真实边界框与感兴趣区域的最大 IoU 值是否大于等于 0.5。如果是，则认为该感兴趣区域与该真实边界框有较高的重叠度。
# pos_idx = np.where(ious >= 0.5)：
#
# 这行代码获取了 IoU 值大于等于 0.5 的感兴趣区域的索引。
# temp_i[0, pos_idx] = ious[pos_idx] 和 temp_i[0] = temp_i[0] / float(temp_i[0].sum())：
#
# 这两行代码将对应索引位置的临时数组 temp_i 赋值为对应的 IoU 值，并对其进行归一化处理。
# box_label[0] += temp_i[0]：
#
# 这行代码将归一化后的临时数组 temp_i 加到关注度标签数组 box_label 中。
# box_label[0] = box_label[0] / num_GT_boxes：
#
# 最后，这行代码将关注度标签数组 box_label 进行进一步归一化处理，以平均每个感兴趣区域上的关注度。
# 返回 box_label：
# 函数最后返回了计算得到的关注度标签数组 box_label。
# 综上所述，该函数实现了根据感兴趣区域与真实边界框之间的 IoU 值计算关注度标签的功能，用于在目标检测任务中指导模型更关注与真实边界框重叠度较高的感兴趣区域。
def compute_targets(ex_rois, gt_rois, query_label):
    """Compute bounding-box regression targets for an image."""
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)  #计算与ex_rois有最大IOU的GT的偏移量 

    targets = ((targets - np.array((0.0, 0.0, 0.0, 0.0)))
            / np.array((0.1, 0.1, 0.2, 0.2)))
    query_bbox_target_data = np.hstack(
            (query_label[:, np.newaxis], targets)).astype(np.float32, copy=False)
    return query_bbox_target_data

def get_query_bbox_regression_labels( query_bbox_target_data):
    query_label = query_bbox_target_data[:, 0]
    query_bbox_targets = np.zeros((query_label.size, 4), dtype=np.float32)
    inds = np.where(query_label > 0)[0]
    if len(inds) != 0:
        for ind in inds:
            query_bbox_targets[ind, :] = query_bbox_target_data[ind, 1:]
    return query_bbox_targets

def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets


def bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    #这个就是为了保证图像边缘的检测结果得到的框不超过图像大小。主要是x2和y2。一个保障机制，一般都不会发生作用。
    """

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


# vis

def debug_pred(debug_dir, count, qvec, img, gt_bbox, roi, iou,tokenizer,senti_pred,setiment_label,rel_pred,rel_label,pred_score,box_target):
    import cv2
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    
    q_str = []
   
    q_str=tokenizer.convert_ids_to_tokens(qvec)
    q_str = ' '.join(q_str)
    
    save_dir = os.path.join(debug_dir, 'box_pred/'+str(count))
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_dir+'/%.3f'%iou, 'w') as f:
        f.write(' ')
    
    with open(save_dir+'/query.txt', 'w') as f:
        f.write(q_str)
        f.write("\r\n senti_pred:"+str(senti_pred))
        f.write("\r\n senti_label:"+str(setiment_label))
        f.write("\r\n relation_pred:"+str(rel_pred))
        f.write("\r\n relation_label:"+str(rel_label))
        f.write("\r\n pred_score:")
        
        for i in range (len(pred_score)):
            f.write("\r\n"+str(pred_score[i])+"  "+str(box_target[i]))

    pred = img.copy()
    box = gt_bbox.astype(np.int)

    box=box[0]
    cv2.rectangle(pred, (box[0], box[1]), (box[2], box[3]), (0, 255,0), 2) #green
    box = roi.astype(np.int)
    cv2.rectangle(pred, (box[0], box[1]), (box[2], box[3]), (0, 255,255), 2)
    
    cv2.imwrite(save_dir+'/pred.jpg', pred)

