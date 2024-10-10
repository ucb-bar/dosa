# DOSA: Differentiable Model-Based One-Loop Search for DNN Accelerators
In this work, we build a differentiable analytical model to enable mapping-first design space exploration of deep learning accelerator designs. We also apply deep learning to adapt this model to the Gemmini accelerator's RTL implementation.

For more details, please refer to:
- [MICRO'23 DOSA Paper](https://people.eecs.berkeley.edu/~ysshao/assets/papers/dosa-micro2023.pdf)
- [MICRO'23 DOSA Slides](https://people.eecs.berkeley.edu/~ysshao/assets/talks/dosa2023-micro-slides.pdf)

If used for research, please cite DOSA by the following publication:

```BibTex
@inproceedings{
  hong2023dosa,
  title={DOSA: Differentiable Model-Based One-Loop Search for DNN Accelerators},
  author={Charles Hong and Qijing Huang and Grace Dinh and Mahesh Subedar and Yakun Sophia Shao},
  booktitle={IEEE/ACM International Symposium on Microarchitecture (MICRO)},
  year={2023},
  url={https://people.eecs.berkeley.edu/~ysshao/assets/papers/dosa-micro2023.pdf}
}
```

## Installation
Requires `python=3.10`.

### DOSA
On a user machine with Python 3.10, clone DOSA:
```
git clone https://github.com/ucb-bar/dosa.git
```

First, acquire a [Gurobi optimizer license](https://www.gurobi.com/features/academic-named-user-license/) and download it to path of choice **($license_path)**. Next, run the following:
```
export GRB_LICENSE_FILE=($license_path)
cd dosa
pip3 install -e .
```

### Timeloop and Accelergy
Install Timeloop and Accelergy on the user machine. The following dependencies are required
(command provided for Debian-based systems):
```
sudo apt install scons libconfig++-dev libboost-dev libboost-iostreams-dev libboost-serialization-dev libyaml-cpp-dev libncurses-dev libtinfo-dev libgpm-dev git build-essential python3-pip
```

Timeloop and Accelergy are available as submodules of the DOSA repository. Install Accelergy and its plug-ins. Make sure you add CACTI and its executables to your **PATH**.

Within `dosa`:
```
git submodule update --init --recursive
cd accelergy-timeloop-infrastructure/src/accelergy
pip3 install .
cd ../cacti
make
cd ..
mv cacti ~/.local/bin
cd accelergy-cacti-plug-in
pip3 install .
cd ../accelergy-aladdin-plug-in
pip3 install .
cd ../accelergy-table-based-plug-ins
pip3 install .
accelergy
accelergyTables
export PATH=$PATH:~/.local/bin/cacti
```

Install Timeloop and add its executables to your **PATH**:
```
cd ../timeloop/src
ln -s ../pat-public/src/pat
cd ..
scons --accelergy --static -j4
export PATH=$PATH:$(pwd)/build
```

## Running the Experiments
> If you run into errors at any point in this section, check that `accelergy`, `cacti` and `timeloop-model` are accessible in your environment, else add them to your **PATH**.

### Figure 4: Analytical model correlation with Timeloop

On the **user machine**, run the following commands. This will correlate DOSA’s differentiable model against Timeloop for our 10,000 point dataset and store the error plots to “output_dir/error_<metric>.png”.
```
cd ../../..
./fig4.sh
```

### Figure 7: Optimization of Gemmini-TL versus baseline algorithms
In the same environment, run the following script, selecting one workload:
```
./fig7.sh (unet|resnet50|bert|retinanet)
```

This will take several hours to run, per workload, and generate a plot at “output_dir/network_searcher_<workload>log<timestamp >.png”. This corresponds to the plot to Figure 5, but over one run rather than averaged over 5. Results should fall within or close to the confidence bounds of the original plot.

### Figure 8: Comparison to hand-tuned accelerators
Only after running `fig7.sh` for the corresponding workload, run:
```
./fig8.sh (unet|resnet50|bert|retinanet)
```

The plots will be generated at the location "output_dir/arch_compare_<workload>_<timestamp>.png". Since these are based on the results of one run rather than averaged over 5, results here will again vary slightly compared to the original plot.

### Figures 10 and 11: Gemmini-RTL performance prediction accuracy
Run the following script:
```
./fig10.sh
```

This will reproduce the plots in Figures 10 and 11 under "output_dir/predict_<predictor>_<dataset>.png". These plots show the prediction accuracy of the three different predictors on the two datasets of Gemmini-RTL latency, which were previously generated using FireSim.

## FireSim-Based Experiments
First, follow the instructions on the [FireSim website](https://docs.fires.im/en/1.20.1/Getting-Started-Guides/AWS-EC2-F1-Getting-Started/) to create an EC2 manager instance. Complete the steps in the “AWS EC2 F1 Getting Started Guide”. Once you have completed up to and including "Setting up your Manager Instance / Key setup, Part 2" in the FireSim docs, you should have a manager instance set up, with an IP address and key. Use ssh or mosh to log in to the instance. 

Next, in "/home/centos", clone the archived FireSim repository.
```
git clone https://github.com/charleshong3/firesim-dosa.git
```

Run the following, which will initialize dependencies and set up FireSim and Chipyard:
```
cd firesim-dosa
./build-setup.sh
sudo yum install autoconf
source sourceme-f1-manager.sh
firesim managerinit --platform f1
```
> If encountering errors with mirror.centos.org, run below code before re-executing ./build-setup.sh.

```bash
sudo sed -i s/mirror.centos.org/vault.centos.org/g /etc/yum.repos.d/*.repo
sudo sed -i s/^#.*baseurl=http/baseurl=http/g /etc/yum.repos.d/*.repo
sudo sed -i s/^mirrorlist=http/#mirrorlist=http/g /etc/yum.repos.d/*.repo
```

After sourcing, complete the steps in "Setting up your Manager Instance / Completing Setup Using the Manager".

**Note that sourceme-f1-manager.sh must be sourced every time you log in to the instance.**

Finally, get the FPGA image used for this experiment. Go to "firesim-dosa/deploy", and paste into `config_hwdb.yaml` the contents of the file in "built-hwdb-entries/" (there should be one file containing a YAML-formatted entry).

### Figure 12: Optimization of Gemmini-RTL
Now, **move to the AWS EC2 instance** set up with the FireSim fork. To run the full workflow of Figure 12, we would need to train two DNN models, run DOSA (constraining the number of PEs to 16x16), select the mappings with the best predicted performance, evaluate latency with FireSim, then combine with energy numbers from Accelergy. To reduce runtime and work that must be done across both the user machine and EC2 instance, we provide the mappings generated by DOSA during this experiment directly to the evaluator as part of our FireSim fork. To build the software for a given workload and run FireSim, run the following:
```
cd ~/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests
./artifact_script.sh (analytical|both|dnn) (unet|resnet50|bert|retinanet)
```
The first argument to `artifact_script.sh` indicates which of the three latency predictors from the previous section should be used. The second argument indicates the target workload. This script launches FireSim automatically and should take a few minutes to run. Depending on the target workload, FireSim will generate either one or two directories under "deploy/results-workload", for matrix multiplication and/or convolutional layers. Pass the previously selected options, along with the directories (**($result_dir_1)** and potentially **($result_dir_2)**) to the parsing script.
```
cd ~/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests
python parse_results.py
  --pred (analytical|both|dnn)
  --workload (unet|resnet50|bert|retinanet)
  --result ($result_dir_1)
  --result ($result_dir_2)
```

This will update the CSV file located at "gemmini-rocc-tests/ artifact/<predictor>/<workload>.csv". **Copy this file back to the user machine**, to your choice of path **($workload_csv)**. On the user machine, run the following to print out the EDP of the Gemmini default mapper/HW and the EDP of the mappings/HW found by DOSA, all using latency numbers from FireSim. The relative magnitude of the Gemmini default and DOSA EDPs should match those in Figure 12.
```
./fig12.sh (unet|resnet50|bert|retinanet) ($workload_csv)
```

When you are done evaluating, go to the EC2 console and terminate your instance(s).