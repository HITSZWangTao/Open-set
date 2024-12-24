# Open-set
This repository includes code and data for comparing the F1-scores of three extreme value distributions: the Weibull, Gumbel, and Frechet distributions. Specifically, we pretrain two person identification models using our mmWave point cloud dataset and a publicly available dataset called mmGait[1]. We then use the top 20 distances between correctly classified training samples and class centers in the feature space to fit the Weibull, Gumbel, and Frechet distributions, respectively. In the inference stage, the distributions for each class generate a value for each given distance, which is used to adjust and recompute the probability of the sample belonging to each class, including the unknown class. Following the metric calculation method defined in [2], F1-scores across 9 openness levels on our dataset are presented in Table I.

<div align="center">
    <em>Table I. F1-scores from three distributions across 9 openness levels on our dataset.</em><br>
    <img src="Imgs/OurDataset.png" alt="Table I",width="400"/>
</div>

Moreover, F1-scores across 9 openness levels on mmGait dataset are presented in Table II.

<div align="center">
    <em>Table II. F1-scores from three distributions across 9 openness levels on mmGait dataset.</em><br>
    <img src="Imgs/mmGaitDataset.png" alt="Table II",width="400"/>
</div>
