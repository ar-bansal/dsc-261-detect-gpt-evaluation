## deepseek-ai/deepseek-coder-1.3b-instruct

python run.py --output_name 19 --base_model_name deepseek-ai/deepseek-coder-1.3b-instruct --mask_filling_model_name Salesforce/codet5-base --n_perturbation_list 1,10,100 --n_samples 700 --pct_words_masked 0.3 --span_length 2 --chunk_size 64 --dataset codenet_data.csv --dataset_key code --cache_dir cache_dir 
echo "\n"
python run.py --output_name 20 --base_model_name deepseek-ai/deepseek-coder-1.3b-instruct --mask_filling_model_name Salesforce/codet5-base --n_perturbation_list 1,10,100 --n_samples 700 --pct_words_masked 0.5 --span_length 2 --chunk_size 64 --dataset codenet_data.csv --dataset_key code --cache_dir cache_dir 
echo "\n" 
python run.py --output_name 21 --base_model_name deepseek-ai/deepseek-coder-1.3b-instruct --mask_filling_model_name Salesforce/codet5-base --n_perturbation_list 1,10,100 --n_samples 700 --pct_words_masked 0.7 --span_length 2 --chunk_size 64 --dataset codenet_data.csv --dataset_key code --cache_dir cache_dir 
echo "\n" 

python run.py --output_name 22 --base_model_name deepseek-ai/deepseek-coder-1.3b-instruct --mask_filling_model_name Salesforce/codet5-base --n_perturbation_list 1,10,100 --n_samples 700 --pct_words_masked 0.3 --span_length 5 --chunk_size 64 --dataset codenet_data.csv --dataset_key code --cache_dir cache_dir 
echo "\n" 
python run.py --output_name 23 --base_model_name deepseek-ai/deepseek-coder-1.3b-instruct --mask_filling_model_name Salesforce/codet5-base --n_perturbation_list 1,10,100 --n_samples 700 --pct_words_masked 0.5 --span_length 5 --chunk_size 64 --dataset codenet_data.csv --dataset_key code --cache_dir cache_dir 
echo "\n" 
python run.py --output_name 24 --base_model_name deepseek-ai/deepseek-coder-1.3b-instruct --mask_filling_model_name Salesforce/codet5-base --n_perturbation_list 1,10,100 --n_samples 700 --pct_words_masked 0.7 --span_length 5 --chunk_size 64 --dataset codenet_data.csv --dataset_key code --cache_dir cache_dir 
echo "\n" 

python run.py --output_name 25 --base_model_name deepseek-ai/deepseek-coder-1.3b-instruct --mask_filling_model_name Salesforce/codet5-base --n_perturbation_list 1,10,100 --n_samples 700 --pct_words_masked 0.3 --span_length 10 --chunk_size 64 --dataset codenet_data.csv --dataset_key code --cache_dir cache_dir 
echo "\n" 
python run.py --output_name 26 --base_model_name deepseek-ai/deepseek-coder-1.3b-instruct --mask_filling_model_name Salesforce/codet5-base --n_perturbation_list 1,10,100 --n_samples 700 --pct_words_masked 0.5 --span_length 10 --chunk_size 64 --dataset codenet_data.csv --dataset_key code --cache_dir cache_dir 
echo "\n" 
python run.py --output_name 27 --base_model_name deepseek-ai/deepseek-coder-1.3b-instruct --mask_filling_model_name Salesforce/codet5-base --n_perturbation_list 1,10,100 --n_samples 700 --pct_words_masked 0.7 --span_length 10 --chunk_size 64 --dataset codenet_data.csv --dataset_key code --cache_dir cache_dir 
echo "\n" 

