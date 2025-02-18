# Personality Traits in Large Language Models

This project contains all the code necessary to verify the results of the paper
> Serapio-García, G., Safdari, M., Crepy, C., Sun, L., Fitz, S., Romero, P.,
> Abdulhai, M., Faust, A., & Matarić, M. "Personality Traits in Large Language
> Models." ArXiv.org. https://doi.org/10.48550/arXiv.2307.00184

## Getting Started / Installation

Most of the code needed to reproduce the results of the paper are shared here in
the form of Jupyter notebooks. The main notebooks and requirements for running
the paper's psychometric and statistical analyses are located in `analysis`.

Inference scripts and requirements for collecting psychometric test and
downstream task data from HuggingFace and OpenAI models are located in
`inference_scripts`.

The main custom dependency of this project is the PsyBORGS psychometric test
administration framework
(https://github.com/google-research/google-research/tree/master/psyborgs). This
repo comes with its own version of the PsyBORGS code, but if a more up-to-date
version is needed, it can be downloaded from the link above. PsyBORGS-related
package dependencies are specified in the `requirements.txt` file in its root
directory. For any other dependencies, they are `pip install`ed in the notebooks
themselves.

## Data

All the test admininistration sessions - which are input for most of the
experiments in the paper - are stored in the `admin_sessions/` directory. Some
of the data used for visualization is stored in the `figures_data/` directory.
All other data are linked in the main paper and can be found on Google's
open source GCP repository:
(https://storage.googleapis.com/personality_in_llms/index.html).

## Citing this work

Please cite the Arxiv paper referenced above. The Bibtex is
> @misc{serapiosafdari2023personality,
>       title={Personality Traits in Large Language Models},
>       author={Greg Serapio-García and Mustafa Safdari and Clément Crepy and
Luning Sun and Stephen Fitz and Peter Romero and Marwa Abdulhai and Aleksandra
Faust and Maja Matarić},
>       year={2023},
>       eprint={2307.00184},
>       archivePrefix={arXiv},
>       primaryClass={cs.CL}
> }

## License and disclaimer

Copyright 2025 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
