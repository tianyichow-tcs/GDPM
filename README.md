
# Modeling the Impact of Timeline Algorithms on Opinion Dynamics Using Low-rank Updates (WWW 2024)

## Gradient descent polarization minimization (GDPM)
GDPM is a gradient descent-based polarization minimization algorithm used to reduce online media polarization by finding the optimal user-interest matrix given under a limited budget.

## Datasets

### Twitter data

We release two anonymized graph datasets, TwitterSmall and TwitterLarge, with ground-truth opinions, user interest and influence.  TwitterSmall contains more than 1000 nodes and TwitterLarge has more than 27,000 nodes (the previously largest publicly available dataset contains less than 550 nodes). For details of data collection and processing, please check Appendix B.1 in the [paper](THIS IS A LINK TO ARXIV).

The datasets are in `main/data/`.  `TwitterLarge.txt` (`TwitterSmall.txt`) is the anonymized edge list. We provide 2 data formats `.jld` and `.csv` in the dataset. The code is written in Julia and it takes `.jld` as default data format. The `s.jld` and `opinion.csv` are the ground-truth opinions. `X.jld` and `user_interest.csv` are the user-to-topic matrices that encode users' interest information towards topics. `Y.jld` and `topic_influence.csv` is the topic-influence matrix that encodes users' influence scores among topics.  The two matrices are both row-stochastic.

The following table summarizes the dataset in two data formats.
| Julia format | CSV format |  Description |
|----------|----------|----------|
| s.jld | opinion.csv| innate opinion data, vector |
| X.jld | user_interest.csv | user intesest matrix |
| Y.jld | topic_influence.csv | topic influencer matrix|


### Realworld dataset
Real-world datasets are publicly available from [Network Data Repository](https://networkrepository.com/). Please check Appendix B.1 for details about how to generate synthetic opinion, user-interest and topic-influence matrices. In this project, we only provide  `Advogato.txt` in `/graph_data/` folder. 

### Random data
We provide Erdős–Rényi random graphs for `Convex.jl` experiments. The data are located in `main/data/Random_*`.


## Usage for optimization

Our code is written by Julia 1.7.2. Run the following command to install packages:

`julia packages.jl`

Run the following command to start optimization on two Twitter datasets with ground truth data.

`sh GDPM_BLs.sh`


Run the following command to generate synthetic innate opinion, user-topic matrix, and topic-influence matrix and start optimization on synthetic data, here we use `Advogato` dataset as an example: 

`julia main_simulation.jl Advogato 0.1 10 100 polarized powerlaw`

[Convex.jl](https://jump.dev/Convex.jl/stable/) is a Julia package for Disciplined Convex Programming (DCP). We provide Convex.jl program with [SCS](https://jump.dev/Convex.jl/stable/solvers/) solver to solve our optimization problem on random graphs.  Run the following command to start optimization:

`sh convex_box.sh`

## Output

The output results of experiments are saved in folder `./output`. The approximation error during optimization is saved in `/output/Errors/`. The `/output/XTs/` folder contains user_interest matrix after optimization. The `/output/Opts/` folder contains the intermediate result of polarization-disagreement index during optimization. The `/output/Logs/` folder contains output from  `main_GDPM_BLs.jl` and `main_simulation.jl`. The `/output/figs/` folder contains figures to visualize the aforementioned results. 

## Convert data to CSV
This project uses `.jld` file format to save all output results. In order to use those result  We also provide a tool to convert `.jld` files to `.csv` files. Run the following code:

`julia convert2csv.jl "data/TwitterSmall/Y.jld" "data/TwitterSmall/topic_influence.csv"`


## Citation
