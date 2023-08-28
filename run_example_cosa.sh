WORKLOAD="resnet50"
ARCH="gemmini"
DATE=`date +%F-%H-%M-%S`
export GRB_LICENSE_FILE='/scratch/charleshong/gurobi.lic'

python run.py -wl $WORKLOAD --arch_name $ARCH --arch_file dataset/hw/${ARCH}/arch/arch.yaml --mapper cosa \
    --output_dir data/${ARCH}_${WORKLOAD}_defaultarch_cosamap_${DATE}