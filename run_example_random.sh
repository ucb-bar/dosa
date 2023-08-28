WORKLOADS="mm conv"
ARCH="gemmini"
NUM_ARCH=100
NUM_MAPPINGS=1000
DATE=`date +%F-%H-%M-%S`

WL_PARAMS=""
for WL in $WORKLOADS; do
    WL_PARAMS=" $WL_PARAMS -wl $WL" # create -wl params
done
WL_STRING=${WORKLOADS// /_} # replace space with underscore

cmd="python run.py $WL_PARAMS --arch_name $ARCH --num_arch $NUM_ARCH --num_mappings $NUM_MAPPINGS \
    --output_dir data/${ARCH}_${WL_STRING}_${NUM_ARCH}arch_${NUM_MAPPINGS}map_${DATE}"
echo $cmd
$cmd
