scancel -u zaho
rm -r training_results/*
rm -r reports/*
rm -r wandb/*
git pull
echo -e "\n\n\n\n\n\n\nCleanup complete!"