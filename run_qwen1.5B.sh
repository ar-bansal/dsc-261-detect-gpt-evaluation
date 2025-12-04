## Qwen/Qwen2.5-Coder-1.5B-Instruct

python run.py --output_name 10 --base_model_name Qwen/Qwen2.5-Coder-1.5B-Instruct --mask_filling_model_name Salesforce/codet5-base --n_perturbation_list 1,10,100 --n_samples 700 --pct_words_masked 0.3 --span_length 2 --chunk_size 64 --dataset codenet_data.csv --dataset_key code --cache_dir cache_dir 
echo "\n"
python run.py --output_name 11 --base_model_name Qwen/Qwen2.5-Coder-1.5B-Instruct --mask_filling_model_name Salesforce/codet5-base --n_perturbation_list 1,10,100 --n_samples 700 --pct_words_masked 0.5 --span_length 2 --chunk_size 64 --dataset codenet_data.csv --dataset_key code --cache_dir cache_dir 
echo "\n" 
python run.py --output_name 12 --base_model_name Qwen/Qwen2.5-Coder-1.5B-Instruct --mask_filling_model_name Salesforce/codet5-base --n_perturbation_list 1,10,100 --n_samples 700 --pct_words_masked 0.7 --span_length 2 --chunk_size 64 --dataset codenet_data.csv --dataset_key code --cache_dir cache_dir 
echo "\n" 

python run.py --output_name 13 --base_model_name Qwen/Qwen2.5-Coder-1.5B-Instruct --mask_filling_model_name Salesforce/codet5-base --n_perturbation_list 1,10,100 --n_samples 700 --pct_words_masked 0.3 --span_length 5 --chunk_size 64 --dataset codenet_data.csv --dataset_key code --cache_dir cache_dir 
echo "\n" 
python run.py --output_name 14 --base_model_name Qwen/Qwen2.5-Coder-1.5B-Instruct --mask_filling_model_name Salesforce/codet5-base --n_perturbation_list 1,10,100 --n_samples 700 --pct_words_masked 0.5 --span_length 5 --chunk_size 64 --dataset codenet_data.csv --dataset_key code --cache_dir cache_dir 
echo "\n" 
python run.py --output_name 15 --base_model_name Qwen/Qwen2.5-Coder-1.5B-Instruct --mask_filling_model_name Salesforce/codet5-base --n_perturbation_list 1,10,100 --n_samples 700 --pct_words_masked 0.7 --span_length 5 --chunk_size 64 --dataset codenet_data.csv --dataset_key code --cache_dir cache_dir 
echo "\n" 

python run.py --output_name 16 --base_model_name Qwen/Qwen2.5-Coder-1.5B-Instruct --mask_filling_model_name Salesforce/codet5-base --n_perturbation_list 1,10,100 --n_samples 700 --pct_words_masked 0.3 --span_length 10 --chunk_size 64 --dataset codenet_data.csv --dataset_key code --cache_dir cache_dir 
echo "\n" 
python run.py --output_name 17 --base_model_name Qwen/Qwen2.5-Coder-1.5B-Instruct --mask_filling_model_name Salesforce/codet5-base --n_perturbation_list 1,10,100 --n_samples 700 --pct_words_masked 0.5 --span_length 10 --chunk_size 64 --dataset codenet_data.csv --dataset_key code --cache_dir cache_dir 
echo "\n" 
python run.py --output_name 18 --base_model_name Qwen/Qwen2.5-Coder-1.5B-Instruct --mask_filling_model_name Salesforce/codet5-base --n_perturbation_list 1,10,100 --n_samples 700 --pct_words_masked 0.7 --span_length 10 --chunk_size 64 --dataset codenet_data.csv --dataset_key code --cache_dir cache_dir 
echo "\n" 

