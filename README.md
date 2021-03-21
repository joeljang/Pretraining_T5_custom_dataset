# Pretraining_T5_custom_dataset
Continue Pretraining T5 on custom dataset

Pretrained models from [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) are made available from [Huggingface](https://huggingface.co/transformers/model_doc/t5.html). This is the code for continuing the pretraining phase of T5 on custom dataset. This code follows the same unsupervised pretraining objective followed by the original paper. Details of the T5 style pretraining can be found in the [paper](https://arxiv.org/abs/1910.10683).

In order to run the code, first install the packages from requirements.txt 
~~~
pip install -r requirements.txt
~~~
You also have to install torch that is compatible with your CUDA version from (https://pytorch.org/)

To run the code, run the following default setting:
~~~
python pretrain.py --input_length 128 --output_length 128 --num_train_epochs 1 --output_dir t5_pretraining --train_batch_size 8 --learning_rate 1e-3 --model t5-base
~~~

In order to fine-tune after continuing pre-training on custom dataset, refer to the following references:
- https://towardsdatascience.com/fine-tuning-a-t5-transformer-for-any-summarization-task-82334c64c81
- https://www.youtube.com/watch?v=r6XY80Z9eSA
