## Qwen/Qwen2.5-Coder-0.5B-Instruct

python run.py --output_name 1 --base_model_name Qwen/Qwen2.5-Coder-0.5B-Instruct --mask_filling_model_name Salesforce/codet5-base --n_perturbation_list 1,10,100 --n_samples 700 --pct_words_masked 0.3 --span_length 2 --chunk_size 64 --dataset codenet_data.csv --dataset_key code --cache_dir cache_dir 
echo "\n"
python run.py --output_name 2 --base_model_name Qwen/Qwen2.5-Coder-0.5B-Instruct --mask_filling_model_name Salesforce/codet5-base --n_perturbation_list 1,10,100 --n_samples 700 --pct_words_masked 0.5 --span_length 2 --chunk_size 64 --dataset codenet_data.csv --dataset_key code --cache_dir cache_dir 
echo "\n" 
python run.py --output_name 3 --base_model_name Qwen/Qwen2.5-Coder-0.5B-Instruct --mask_filling_model_name Salesforce/codet5-base --n_perturbation_list 1,10,100 --n_samples 700 --pct_words_masked 0.7 --span_length 2 --chunk_size 64 --dataset codenet_data.csv --dataset_key code --cache_dir cache_dir 
echo "\n" 

python run.py --output_name 4 --base_model_name Qwen/Qwen2.5-Coder-0.5B-Instruct --mask_filling_model_name Salesforce/codet5-base --n_perturbation_list 1,10,100 --n_samples 700 --pct_words_masked 0.3 --span_length 5 --chunk_size 64 --dataset codenet_data.csv --dataset_key code --cache_dir cache_dir 
echo "\n" 
python run.py --output_name 5 --base_model_name Qwen/Qwen2.5-Coder-0.5B-Instruct --mask_filling_model_name Salesforce/codet5-base --n_perturbation_list 1,10,100 --n_samples 700 --pct_words_masked 0.5 --span_length 5 --chunk_size 64 --dataset codenet_data.csv --dataset_key code --cache_dir cache_dir 
echo "\n" 
python run.py --output_name 6 --base_model_name Qwen/Qwen2.5-Coder-0.5B-Instruct --mask_filling_model_name Salesforce/codet5-base --n_perturbation_list 1,10,100 --n_samples 700 --pct_words_masked 0.7 --span_length 5 --chunk_size 64 --dataset codenet_data.csv --dataset_key code --cache_dir cache_dir 
echo "\n" 

python run.py --output_name 7 --base_model_name Qwen/Qwen2.5-Coder-0.5B-Instruct --mask_filling_model_name Salesforce/codet5-base --n_perturbation_list 1,10,100 --n_samples 700 --pct_words_masked 0.3 --span_length 10 --chunk_size 64 --dataset codenet_data.csv --dataset_key code --cache_dir cache_dir 
echo "\n" 
python run.py --output_name 8 --base_model_name Qwen/Qwen2.5-Coder-0.5B-Instruct --mask_filling_model_name Salesforce/codet5-base --n_perturbation_list 1,10,100 --n_samples 700 --pct_words_masked 0.5 --span_length 10 --chunk_size 64 --dataset codenet_data.csv --dataset_key code --cache_dir cache_dir 
echo "\n" 
python run.py --output_name 9 --base_model_name Qwen/Qwen2.5-Coder-0.5B-Instruct --mask_filling_model_name Salesforce/codet5-base --n_perturbation_list 1,10,100 --n_samples 700 --pct_words_masked 0.7 --span_length 10 --chunk_size 64 --dataset codenet_data.csv --dataset_key code --cache_dir cache_dir 
echo "\n" 

