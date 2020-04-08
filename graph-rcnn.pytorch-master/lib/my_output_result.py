import numpy as np


def corresponding_obj_rel(dataset,predictions,predictions_pred):
    write_result = open('my_result.txt', 'w')
    write_result_att = open('my_groudtruth.txt', 'w')

    for image_id, (prediction, prediction_pred) in enumerate(zip(predictions, predictions_pred)):
        # print(image_id, (prediction, prediction_pred) )
        # break

        img_info = dataset.get_img_info(image_id)
        gt_boxlist = dataset.get_groundtruth(image_id)
        ground_truth_attr=gt_boxlist.get_field("atts").numpy()
        ground_truth_label=gt_boxlist.get_field("labels").numpy()
        image_width = img_info["width"]
        image_height = img_info["height"]
        # import pdb; pdb.set_trace()
        prediction = prediction.resize((image_width, image_height))
        obj_scores = prediction.get_field("scores").numpy()
        all_rels = prediction_pred.get_field("idx_pairs").numpy()
        fp_pred = prediction_pred.get_field("scores").numpy()

        scores = np.column_stack((
            obj_scores[all_rels[:, 0]],
            obj_scores[all_rels[:, 1]],
            fp_pred[:, 1:].max(1)
        )).prod(1)
        sorted_inds = np.argsort(-scores)
        sorted_inds = sorted_inds[scores[sorted_inds] > 0]  # [:100]
        category=dataset.ind_to_classes
        rel_category=dataset.ind_to_predicates
        attr_category=dataset.ind_to_att
        label= prediction.get_field("labels").numpy()
        atts= prediction.get_field("atts").numpy()
        mid_label=[]
        label = [category[i] for i in label]
        ground_truth_label = [category[i] for i in ground_truth_label]
        # print(len(atts),len(label),len(ground_truth_attr),len(ground_truth_label))
        write_result_att.write(str(image_id)+'\n')
        for no, i in enumerate(ground_truth_label):
            if ground_truth_attr[no] != 0:
                gr_attr = attr_category[ground_truth_attr[no]]
            else:
                gr_attr = ''
            if i.find('background') == -1:
                if i in mid_label:
                    count = 0
                    while (i + str(count) in mid_label):
                        count += 1
                    write_result_att.write(gr_attr + ' ' + i + str(count) + '\n')
                    mid_label.append(i + str(count))
                else:
                    write_result_att.write(gr_attr + ' ' + i + '\n')
                    mid_label.append(i)
        mid_label=[]
        final_label=[]
        for no, i in enumerate(label):
            if atts[no]!=0:
                attrs=attr_category[atts[no]]
            else:
                attrs=''
            if i.find('background')==-1:
                if i in mid_label:
                    count=0
                    while(i+str(count) in mid_label):
                        count+=1
                    final_label.append(attrs+' '+i+str(count))
                    mid_label.append(i+str(count))
                else:
                    final_label.append(attrs+' '+i)
                    mid_label.append(i)
            else:
                final_label.append('')
        rel_pair=all_rels[sorted_inds],
        rel=np.argmax(fp_pred[sorted_inds],axis=1)
        rel = [rel_category[i] for i in rel]
        write_result.write(str(image_id)+'\n')
        for itr_attr in range(len(final_label)):
            if  final_label[itr_attr]!='':
                write_result.write(final_label[itr_attr]+'\n')
        write_result.write('relationship----------------------------------------------------'+'\n')
        for itr_rel in range(len(rel_pair[0])):
            if rel[itr_rel].find('__background__')==-1 and len(final_label[rel_pair[0][itr_rel][0]])!=0 and len(final_label[rel_pair[0][itr_rel][1]])!=0:
                write_result.write(
                final_label[rel_pair[0][itr_rel][0]] + '-' +final_label[rel_pair[0][itr_rel][1]] + ' ' + rel[
                    itr_rel] + '\n')
