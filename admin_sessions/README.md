This directory contains all the admin sessions used in the various experiment runs in the paper. These can be considered the "input" to the experiments.

Admin sessions are defined here: https://github.com/google-research/google-research/blob/master/psyborgs/survey_bench_lib.py#L70 and are part of the PsyBORGS open source framework (https://github.com/google-research/google-research/tree/master/psyborgs).

More details of the admin sessions:

* prod_run_01_external_rating.json: Used for the construct validity experiments on GPT, Llama, Mistral, Mixtral models.
* ablation01_ind_big5_9lvls_50desc_admin_session_rating.json: Used for single trait shaping experiments on GPT, Llama, Mistral, Mixtral models.
* ablation_03_conc_big5_2lvls_50desc_admin_session_rating.json: Used for multi trait shaping experiments on GPT, Llama, Mistral, Mixtral models.
* generate_updates_ablation_01_admin_session_25.json: Used for the real-world task (social media update generation) experiment on PaLM, GPT, Llama, Mistral, Mixtral models.

* palm_prod_run_01_numeric_personachat_admin_session.json: Used for the construct validity experiments on the PaLM models.
* palm_ablation_01_ind_5doms_9lvls_50desc_admin_session.json: Used for single trait shaping experiments on PaLM models.
* palm_ablation_03_conc_big5_2lvls_50desc_admin_session.json: Used for multi trait shaping experiments on PaLM models.
