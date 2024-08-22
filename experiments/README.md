## Contributions

We proved that the SAE features related to the finetuning target will change greater than other features.

We developed an automated model audit method based on developmental interpretability: 



## Our Method


This automated audit tool will directly investigate the dictionary of a given prompt set and calculate the ratio of features that shifted during the finetune process. If, compared to general feature sets, the feature shifted greatly, then, without testing what the outputs actually are, we can figure out that the model is finetuned on the domain related to the dataset.


## Experiments

To ensure the validity of our experiments and to encompass a diverse range of tasks, model architectures, and model sizes, we selected various tasks including safety, math, reasoning, empathy and language adapting as our target tasks and chose Qwen-1.1-0.5B-chat, Gemma-2B-it, and Mistral-7B-Instruct-v0.1 as our target models for conducting the experiments. We finetuned the target model in specific domains and test their top-k feature intersection before / after the finetuning on task specific and general alpaca dataset. The results are shown in the figure below.

Observations: On each task, the intersection mean on the task specific dataset is lower than that on the general dataset. This indicates that the model's features related to the specific task are changed greater after the targeted finetuning. Moreover, the intersection mean difference rate is bigger than 15%, suggesting that this change is significant.

## Ablation

Moreover, we analyze the effect of different top-k values on the intersection mean difference. The results are shown in the figure above.

Observations: The intersection mean difference rate decreases with the increase of top-k values, and converges to a constant value when the top-k value is high. Similar phenomenon is also obeserved in other models, although detailed values are different. This indicates that our methods' performance is not constrained by a large top-k value.

## Limitation and Future Work

In this summer, we only confirm that the SAE features related to the finetuning target will change greater than other features. However, we do not have a thorough analysis on the mechanism behind this phenomenon. We will conduct more experiments to understand the reasons behind this phenomenon in the future.