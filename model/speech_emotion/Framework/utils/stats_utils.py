from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt


def get_float(value):
    try:
        new_val = float(value)
    except ValueError:
        new_val = 0
    return new_val


def generate_graphs(csv_path, save_dir):
    matplotlib.rcParams.update({'font.size': 50})
    labels = defaultdict(lambda: defaultdict(list))
    data = defaultdict(lambda: defaultdict(list))
    meta = defaultdict(lambda: defaultdict(list))
    first_row = True
    split_index = 1
    with open(csv_path, "r") as fp:
        for row in fp:
            if first_row:
                # Find the index for the split
                column_split = row.strip().split(",")
                for i, column in enumerate(column_split):
                    if column == 'data_split':
                        split_index = i
                        break
                first_row = False
                continue
            row_split = row.strip().split(",")
            data_split = row_split[split_index]
            epoch = get_float(row_split[split_index + 1])
            iters = get_float(row_split[split_index + 2])
            acc = get_float(row_split[split_index + 3])
            macro = get_float(row_split[split_index + 4])
            au_roc_macro = get_float(row_split[split_index + 5])
            au_roc_micro = get_float(row_split[split_index + 6])
            au_prc_macro = get_float(row_split[split_index + 7])
            au_prc_micro = get_float(row_split[split_index + 8])
            losses = [get_float(loss) for loss in row_split[split_index + 9].split(' ')]

            labels['epochs'][data_split].append(epoch)
            labels['iterations'][data_split].append(iters)
            data['accuracy'][data_split].append(acc)
            data['macro_f1'][data_split].append(macro)
            data['au_roc_macro'][data_split].append(au_roc_macro)
            data['au_roc_micro'][data_split].append(au_roc_micro)
            data['au_prc_macro'][data_split].append(au_prc_macro)
            data['au_prc_micro'][data_split].append(au_prc_micro)
            data['losses'][data_split].extend(losses)
            meta['num_of_losses'][data_split].append(len(losses))

    for datatype in data:
        y_lim = (0, 1)
        for label in labels:
            plt.figure(figsize=(80, 45))
            for key in labels[label]:
                x = labels[label][key]
                y = data[datatype][key]
                if datatype == 'losses':
                    # Update y limits
                    curr_y_lim = (min(y), max(y))
                    y_lim = (min(curr_y_lim[0], y_lim[0]), max(curr_y_lim[1], y_lim[1]))
                    # one loss for each iteration
                    if label == 'iterations':
                        x = range(len(y))
                    elif label == 'epochs':
                        shrink_y = []
                        extended_x = []
                        old_y = 0
                        for epoch in x:
                            num_losses_in_epoch = meta['num_of_losses'][key][int(epoch)]
                            y_position = old_y + num_losses_in_epoch
                            vals_to_avg = y[old_y:y_position]
                            shrink_y.append(sum(vals_to_avg)/len(vals_to_avg))
                            # extended_x.extend([epoch]*meta['num_of_losses'][key][int(epoch)])
                        # x = extended_x
                        y = shrink_y
                elif label == 'epochs':
                    # Average over the values calculated in the same epoch
                    new_x = []
                    new_y = []
                    ys_to_avg = []
                    curr_epoch = None
                    for i, epoch in enumerate(x):
                        if curr_epoch is None:
                            curr_epoch = epoch
                            ys_to_avg.append(y[i])
                            continue
                        if curr_epoch == epoch:
                            ys_to_avg.append(y[i])
                            continue
                        if curr_epoch != epoch:
                            avg_ys = sum(ys_to_avg)/len(ys_to_avg)
                            new_y.append(avg_ys)
                            new_x.append(curr_epoch)
                            ys_to_avg = [y[i]]
                            curr_epoch = epoch
                    x = new_x
                    y = new_y

                print('Plotting {} ({}) against {} ({}) for split {}'.format(datatype, len(y), label, len(x), key))
                plt.plot(x, y, label=key, linewidth=3.0)
            plt.ylim(y_lim[0], y_lim[1])
            plt.ylabel(datatype)
            plt.xlabel(label)
            plt.legend(loc="upper left")
            for _, spine in plt.gca().spines.items():
                spine.set_linewidth(6)
            plt.savefig(save_dir + '/{} against {}.png'.format(datatype, label))
            plt.close()


# generate_graphs('/home/james/models/joint/3_emotions_stats.csv', '/home/james/models/joint/3_emotion')
# generate_graphs('/home/james/models/joint/stats.csv', '/home/james/models/joint/4_emotion')
# generate_graphs('/home/james/models/joint/3_emotions_stats_normalised.csv', '/home/james/models/joint/3_emotion_normalised')
# generate_graphs('/home/james/models/multihead/full_emotion_detail_stats_with_normalisation.csv', '/home/james/models/multihead/graphs_with_normalisation')
# generate_graphs('/home/james/models/multihead/all_emotion_detailed_stats_weighted_class_loss.csv', '/home/james/models/multihead/single_model_graphs')
# generate_graphs('/home/james/models/multihead/multi_model_emotion_detailed_stats_weighted_class_loss.csv', '/home/james/models/multihead/multi_model_graphs')
# generate_graphs('/home/james/checkpoints/temp_nlp_bert-base-uncased.csv', '/home/james/checkpoints/bert_base')
# generate_graphs('/home/james/checkpoints/temp_nlp_bert-large-uncased.csv', '/home/james/checkpoints/bert_large')
# generate_graphs('/home/james/checkpoints/temp_nlp_distilbert-base-uncased.csv', '/home/james/checkpoints/distilbert')
# generate_graphs('/home/james/checkpoints/multihead/all_emotion_detailed_stats_weighted_class_loss.csv', '/home/james/checkpoints/multihead/graphs')
# generate_graphs('/home/james/checkpoints/multimodel/multi_model_emotion_detailed_stats_weighted_class_loss.csv', '/home/james/checkpoints/multimodel/graphs')
generate_graphs('/home/james/checkpoints/multihead_final/mid_fusion.csv', '/home/james/checkpoints/multihead_final/mid')
generate_graphs('/home/james/checkpoints/multihead_final/mid_fusion_attention_heads.csv', '/home/james/checkpoints/multihead_final/att_head')
