def parse_cfg(cfgfile):
    fp = open(cfgfile, 'r')
    line = fp.readline()
    block = dict()
    training_data_list = []
    while line != '':
        line = line.rstrip()
        if line == '' or line[0] == '#':
            line = fp.readline()
            continue        
        else:
            key,value = line.split('=')
            key = key.strip()
            if key == 'type':
                key = '_type'
            value = value.strip()

            if key.lower().endswith("_weight") or key.lower() in ['train_with_gt_codes_dropout', 'auto_gt_codes_dropout_bias',
                                                                  'network_leaky_slope', 'network_graph_leaky_slope',
                                                                  'init_network_graph_leaky_slope', 'conf_factor_tau',
                                                                  'conf_network_leaky_slope', 'conf_network_graph_leaky_slope']:
                value = float(value)
            elif value.isnumeric():
                value = int(value)

            if key.startswith('learning_rate') or key in ['padding_ratio', 'train_obj_visible_theshold',
                                                          'second_dataset_ratio', 'vert_visib_ratio', 'change_bg_prob']:
                value = float(value)

            if value == 'False':
                value = False
            elif value == 'True':
                value = True
                
            block[key] = value
        line = fp.readline()

    fp.close()
    return block