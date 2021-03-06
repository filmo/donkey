import pickle
from pprint import pprint
import numpy as np
import os
import glob

exp_defs = []
# from testing_files.exp_13_aug import e_def as exp_n
# exp_defs.append(exp_n([]))
# from testing_files.exp_13a import e_def as exp_n
# exp_defs.append(exp_n([]))
# from testing_files.exp13b import e_def as exp_n
# exp_defs.append(exp_n([]))
# from testing_files.exp14 import e_def as exp_n
# exp_defs.append(exp_n([]))
from testing_files.training_experiments.exp16_shuf import e_def as exp_n
exp_defs.append(exp_n([]))
from testing_files.training_experiments.exp17b import e_def as exp_n
exp_defs.append(exp_n([]))
from testing_files.training_experiments.exp17 import e_def as exp_n
exp_defs.append(exp_n([]))



np.set_printoptions(precision=5, suppress=True, linewidth=150)

pkl_path = '../models/2018-07-31_shuffled'

pattern = '*.pkl'

# gather all the individual pickle histories 
all_pkl_files = glob.glob(os.path.join(pkl_path,pattern))


all_hist = {}
for exp in exp_defs:
    for e in exp:
        all_hist[e['exp']] = e

hist_order = ['loss', 'val_loss', 'out_0_loss', 'out_1_loss', 'val_out_0_loss', 'val_out_1_loss']

overall_best = 9999
overall_winner = {}
all_result_rows = []
all_avg_rows = []

exp_data = []
avg_data = {}

for gpu_id_exp in all_pkl_files:
    file = open( gpu_id_exp, 'rb')
    exp_hist_gpu = pickle.load(file)

    for _, (exp_num, v) in enumerate(exp_hist_gpu.items()):
        if not exp_num in avg_data:
            avg_data[exp_num] = []
        for _, (batch_size, v) in enumerate(v.items()):
            # if batch_size != 128:
            #     continue
            run_data_init = [exp_num, batch_size]
            for _, (run_num, results) in enumerate(v['run'].items()):
                # if results['aug'] == True:
                #     continue
                # build the row of results for this particular run of the experiment 
                run_data = run_data_init + [run_num, results['aug']] + results['best_epoch'].tolist()
                # add it to the global tracker
                exp_data.append(run_data)
                # gather all the runs (by batch and run number) into an average list 
                avg_data[exp_num].append(results['best_epoch'].tolist() + [results['aug']])
            # because the experiments are spread across separate GPU and pickle files we have to first aggregate them
# all before doing the overall averaging. 

averaged_results = []
for _, (exp_num, data) in enumerate(avg_data.items()):
    data_np = np.asarray(data)
    if len(data_np) == 0:
        continue
    avg = np.mean(data_np, axis=0)
    std = np.std(data_np,axis=0)
    averaged_results.append([exp_num] + avg.tolist() + [std[1].tolist()])

# exp_data is all the individual runs with batch_size, run_#, and Aug Flag 
exp_data_np = np.asarray(exp_data)
# sort the table by experiment number 
sorted_data = exp_data_np[exp_data_np[:, 5].argsort()]

print('Experiments analyized:', len(averaged_results), 'Total Runs:', len(sorted_data))
print('All Experimental results sorted best to worst')
for e in sorted_data:
    print('%25s\tbs:%d \tAug:%d\tVal: %2.5f\tStd: %1.5f' %(all_hist[e[0]]['model_base_name'],
                                          e[1],
                                          e[3],
                                          e[5],
                                          e[9]))

avg_data_np = np.asarray(averaged_results)
sorted_avg = avg_data_np[avg_data_np[:, 2].argsort()]
aug_text = {0:'False',0.5:'Mixed',1:'True'}
print('\n\nAverge per experiement sorted best to worst')
for e in sorted_avg:
    print('Exp #%2d %25s \tAug:%6s\tVal: %2.5f\tStd: %2.5f\tImprovement: %2.4f %% ' % (e[0],all_hist[e[0]]['model_base_name'],
                                                                                       aug_text[e[7]],
                                                e[2],
                                                e[8],
                                                (abs(e[2]-sorted_avg[-1,2])/sorted_avg[-1,2])*100))


print("\n")
winning_exp_num = sorted_data[0, 0]
print('Best overall Experiment: #%i' % winning_exp_num, ' File Name:', all_hist[winning_exp_num]['model_base_name'])
print('Min Val Loss: %2.5f Batch Size: %i' % (sorted_data[0, 5], sorted_data[0, 1]))

print('\nTop 3 models on average:')
for i in range(3):
    print('Experiment %i' % sorted_avg[i, 0], 'File Name: %25s' % all_hist[sorted_avg[i, 0]]['model_base_name'],
          '\t\tAug:', all_hist[sorted_avg[i, 0]]['aug'], 'loss: %2.5f' % sorted_avg[i, 2])

print('\nBottom 3 models on average:')
for i in range(2,-1,-1):
    print('Experiment %i' % sorted_avg[-1 - i, 0], 'File Name:', all_hist[sorted_avg[-1 - i, 0]]['model_base_name'],
          '\t\tAug:', all_hist[sorted_avg[-1 - i, 0]]['aug'], 'loss: %2.5f ' % sorted_avg[-1 - i, 2])

print('\nPickle Files:')
pprint(all_pkl_files)