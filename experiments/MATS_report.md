# Final Report: Automating LLM Auditing with Developmental Interpretability

*Produced as part of the* [ML Alignment & Theory Scholars Program](https://www.matsprogram.org/) *- Summer 2024 Cohort, supervised by Evan Hubinger*

# TL: DR

- We performed experiments which proved that the SAE features related to the finetuning target will change greater than other features in the semantic space.
- We developed an automated model audit method based on this finding.
- We investigated the robustness of this method on different data, models, and hyper-parameters.

# 1. Introduction

**The openness of powerful models brings risks.** Currently, an increasing number of powerful models are being opened up through various means including open-source weights and offering APIs for fine-tuning, in order to meet users' customized needs. (Just when I was writing the draft of this post, I heard that OpenAI made finetuning GPT-4o available through API.) This process provides many potentially malicious users with opportunities to [inject toxic content](https://arxiv.org/pdf/2405.08597) or backdoor switches, which is namely [the challenge of open-source governance](https://arxiv.org/abs/2310.19852).

**Evaluating frontier AI risk is challenging and costly.** As model capabilities scale up, the potential security risks and failure modes for humans become increasingly complex. This is particularly true with recent research findings such as [backdoor injection](https://arxiv.org/abs/2401.05566), [deceptive alignments](https://www.pnas.org/doi/pdf/10.1073/pnas.2317967121), and other security risks. Although some scholars have begun exploring the [evaluation](https://arxiv.org/pdf/2407.00948) of these complex risks, it is clear that as the model capabilities scale up, it becomes increasingly difficult to directly assess the security risks from just the output content.

**Insufficient model audit leads to failure.** Given the risks stated in the paragraphs above, along with the decreasing difficulty of obtaining GPU power for model fine-tuning or continued pre-training, we can identify a potentially high-risk failure mode: Among the huge quantity of fine-tuned models, only a few have unsafe content injected, and the cost and time for conducting a comprehensive, fine-grained evaluation on these models (as an audit method) are extremely high. When only general evaluation tasks are performed in the [audit game](https://www.lesswrong.com/posts/cQwT8asti3kyA62zc/automating-auditing-an-ambitious-concrete-technical-research), some models with hidden backdoors can bypass this test and are deployed publicly, causing significant negative impacts, including but not limited to the activation of these backdoors, the spread of inappropriate values, or the models beginning to deceive users, leading to potential takeovers.

**Developmental Interpretability.** 

Developmental interpretability is an emerging subfield of mechanistic interpretability that focuses on understanding how models evolve and learn certain features from the dataset during training or fine-tuning processes ([Olsson, C., et al.](https://arxiv.org/abs/2209.11895), [Nanda, N., et al.](https://arxiv.org/abs/2301.05217)). Previous works on developmental interpretability mainly focused on a small amount of parameters and features. Thus, I'm curious of the macro mechanism behind the finetune process: 

*What happens to the interpretable features during training / finetuning in statistical terms, and can we use these dynamics to audit the model?*

Specifically, in this project, I intend to investigate the finetuning process using mech interp tools (specifically, SAE) to find and analyze how interpretable features move in semantic space after finetuning.

# 2. Methodology

Basically, our method involves comparing the feature movement of the models before and after finetuning on task-specific and general datasets. We picked top-k SAE features toward each token in the dataset and then calculate the feature changes of these top-k features before and after finetuning, denoted as $\mathcal{N}$. Using this method, we acquire $\mathcal{N}$ in the task-specific dataset and general dataset, denoted as $\mathcal{N}_s$ and $\mathcal{N}_b$, respectively. Last, we calculate the intersection of $\mathcal{N}_s$ and $\mathcal{N}_b$ to get the Intersection Difference Rate, which is used to measure the change of feature distribution related to the specific task. The bigger the Intersection Difference Rate is, the more SAE features related to the specific task are changed, and this suggests that the model has learned on the specific task more.

The feasibility and advantages of my approach compared to other methods in automated audit are explained below:
- Low cost, high automation: The prompt set is fixed for a given model, and the shift is directly calculated in terms of ratio, so there’s no need to evaluate the result using GPT-4 or human annotators. 
- Stability: The shift ratio calculation is more stable than evaluating the output since the temperature will easily affect the result of the model generation, making these methods more stable than traditional evaluation.
- Generalizable across multiple models and tasks: Also, this method is highly generalizable across models and tasks. By extracting a set of features using SAE, any given model can be audited in this way. 


# 3. Experiments

To ensure the validity of our experiments, we selected various tasks as our target tasks and chose Qwen-1.1-0.5B-chat, Gemma-2B-it, and Mistral-7B-Instruct-v0.1 as our target models for conducting the experiments. We finetuned the target model in specific domains and test their top-k feature intersection before / after the finetuning on task specific and general alpaca dataset. 

Observations: On each task, the intersection mean on the task specific dataset is lower than that on the general dataset. This indicates that the model's features related to the specific task are changed greater after the targeted finetuning. Moreover, the intersection mean difference rate is bigger than 15%, suggesting that this change is significant.
