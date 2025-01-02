cd ../../om/user/zaho/bfm_ic/
git pull

srun -n 1 --gres=gpu:a100:1 -t 16:00:00 --mem 40G --pty bash