# TStruct: Learning Shared Temporal Structures for Long-Term Time-Series Forecasting (NeurIPS 2026)

[OpenReview](https://openreview.net/forum?id=q3fXT85cph) · [arXiv](https://arxiv.org/abs/2610.17864)

## Motivation

Distribution shifts and cross-domain heterogeneity make it challenging to learn generalizable representations directly from time-series observations. We hypothesize that, despite differences in their observed values, real-world time series reflect underlying temporal evolution governed by shared relational structures. To investigate this hypothesis, we represent temporal relationships using patch-level dependency matrices and analyze their spectral properties across samples and variables. The observed spectral concentration motivates modeling temporal dependencies through a shared low-dimensional structure basis. Building on this insight, TStruct learns shared temporal structures and adaptively combines them for long-term forecasting.

<p align="center">
  <a href="figures/motivation.png"><img src="figures/motivation.png" alt="Motivation: shared temporal structures across samples and variables" width="1000"></a>
</p>

## Method

TStruct directly parameterizes shared temporal structures, using appropriate initialization and limiting their flexibility to mitigate overfitting. Even simple adaptive combinations of these structures can yield strong performance in long-term forecasting. A parameter-efficient channel adaptation head maps the resulting representations to future predictions.

<p align="center">
  <a href="figures/methodology.png"><img src="figures/methodology.png" alt="TStruct architecture, Temporal Structure Mixer, and PCAHead" width="1000"></a>
</p>

## Time Structure Visualizations

We provide additional visualizations of the learned temporal structures across all benchmark datasets. Red and blue indicate higher and lower relative aggregation preferences, respectively. The visualizations show that the learned structures are not random mixing patterns.

<details open>
<summary><strong>ETTh1</strong></summary>

<p align="center">
  <a href="figures/ETTh1.png"><img src="figures/ETTh1.png" alt="Learned temporal structures on ETTh1" width="1000"></a>
</p>

</details>

<details>
<summary><strong>ETTh2</strong></summary>

<p align="center">
  <a href="figures/ETTh2.png"><img src="figures/ETTh2.png" alt="Learned temporal structures on ETTh2" width="1000"></a>
</p>

</details>

<details open>
<summary><strong>ETTm1</strong></summary>

<p align="center">
  <a href="figures/ETTm1.png"><img src="figures/ETTm1.png" alt="Learned temporal structures across encoder layers on ETTm1" width="1000"></a>
</p>

</details>

<details>
<summary><strong>ETTm2</strong></summary>

<p align="center">
  <a href="figures/ETTm2.png"><img src="figures/ETTm2.png" alt="Learned temporal structures across encoder layers on ETTm2" width="1000"></a>
</p>

</details>

<details>
<summary><strong>Weather</strong></summary>

<p align="center">
  <a href="figures/Weather.png"><img src="figures/Weather.png" alt="Learned temporal structures across encoder layers on Weather" width="1000"></a>
</p>

</details>

<details open>
<summary><strong>Electricity</strong></summary>

<p align="center">
  <a href="figures/Electricity.png"><img src="figures/Electricity.png" alt="Learned temporal structures across encoder layers on Electricity" width="1000"></a>
</p>

</details>

<details>
<summary><strong>Traffic</strong></summary>

<p align="center">
  <a href="figures/Traffic.png"><img src="figures/Traffic.png" alt="Learned temporal structures across encoder layers on Traffic" width="1000"></a>
</p>

</details>

## Results

<p align="center">
  <a href="figures/results.png"><img src="figures/results.png" alt="Forecasting performance comparison of TStruct and baseline models" width="1000"></a>
</p>

## Getting Started

### 1. Prepare the Datasets

We provide training and evaluation scripts for the following public datasets:

**ETTh1, ETTh2, ETTm1, ETTm2, Weather, Electricity, and Traffic.**

📥 Download from [Google Drive](https://drive.google.com/file/d/1ypgCc6iQ2Z8IB_9CY3If_KMRNQKBsI3J/view?usp=sharing).

Create a `datasets` directory in the project root:

```bash
mkdir -p datasets
```

Extract the downloaded files and arrange the CSV files as follows:

```text
TStruct-main/
├── datasets/
    ├── ETTh1.csv
    ├── ETTh2.csv
    ├── ETTm1.csv
    ├── ETTm2.csv
    ├── weather.csv
    ├── electricity.csv
    ├── traffic.csv
    └── your_dataset.csv
```

### 2. Run the Experiments

Run the following command from the project root to train and evaluate TStruct on ETTh1:

```bash
bash scripts/TStruct/etth1.sh
```

Scripts for the other datasets are available in `scripts/TStruct/`:

```bash
bash scripts/TStruct/etth2.sh
bash scripts/TStruct/ettm1.sh
bash scripts/TStruct/ettm2.sh
bash scripts/TStruct/weather.sh
bash scripts/TStruct/electricity.sh
bash scripts/TStruct/traffic.sh
```

After creating and configuring your custom dataset script, run:

```bash
bash scripts/TStruct/your_dataset.sh
```

Evaluation results are appended to `result.txt`.

## Citation

If you find this work helpful, please consider citing it:

```bibtex
@inproceedings{lyz2026tstruct,
  title={{TS}truct: Learning Shared Temporal Structures for Long-Term Time-Series Forecasting},
  author={Yangze Li and Zhengyang Zhou and Qihe Huang and Pengkun Wang and Binwu Wang and Yang Wang},
  booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
  year={2026},
  url={https://openreview.net/forum?id=q3fXT85cph}
}
```

## Acknowledgements

We would like to thank the authors of the following open-source projects for their valuable contributions, which have provided significant help for our work:

- [LTSF-Linear](https://github.com/cure-lab/LTSF-Linear)
- [OpenTS](https://decisionintelligence.github.io/OpenTS/)


## Contact

If you have any questions, please contact [liyangze@mail.ustc.edu.cn](mailto:liyangze@mail.ustc.edu.cn) or [submit an issue](https://github.com/yangzeli/TStruct/issues).

