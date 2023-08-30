import subprocess
# getting fids for celeba without mild, mild with momentum and mild with nesterov
# T = [20, 30, 40, 50]
T = [1]
gammas = [0.05, 0.07]
for t in T:
    for gamma in gammas:
        #--------------------------------FID Calculations-----------------------------------------
        # with nesterov i.e MILD
        subprocess.call(['python', 'main.py', '--T', str(t), '--doc', 'celeba', '--find_fid', '-o', 'sample_69/celeba/nesterov', 
                         '--gamma', str(gamma), '--step_size_gamma', '0.02'])
        # with momentum i.e. MILD
        subprocess.call(['python', 'main.py', '--T', str(t), '--doc', 'celeba', '--find_fid', '-o', 'sample_69/celeba/momentum', 
                         '--gamma', str(gamma), '--step_size_gamma', '0.02', '--mild_momentum'])
        # without momentum i.e. without MILD
        subprocess.call(['python', 'main.py', '--T', str(t), '--doc', 'celeba', '--find_fid', '-o', 'sample_69/celeba/without_mild', 
                         '--without_mild'])

        #-------------------------------------TEST---------------------------------------------------

        # # with nesterov i.e MILD
        # subprocess.call(['python', 'main.py', '--T', str(t), '--doc', 'celeba', '--test', '-o', 'sample_69/celeba/nesterov', 
        #                  '--gamma', str(gamma), '--step_size_gamma', '0.02'])
        # # with momentum i.e. MILD
        # subprocess.call(['python', 'main.py', '--T', str(t), '--doc', 'celeba', '--test', '-o', 'sample_69/celeba/momentum', 
        #                  '--gamma', str(gamma), '--step_size_gamma', '0.02', '--mild_momentum'])
        # # without momentum i.e. without MILD
        # subprocess.call(['python', 'main.py', '--T', str(t), '--doc', 'celeba', '--test', '-o', 'sample_69/celeba/without_mild', 
        #                  '--without_mild'])
