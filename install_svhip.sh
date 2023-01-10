#!/bin/bash

THIS_DIR=$(basename $(pwd))
cd ..
mkdir -p $CONDA_PREFIX/share
cp -r $THIS_DIR $CONDA_PREFIX/share
mkdir -p $CONDA_PREFIX/bin/
echo "#!/bin/sh" > $CONDA_PREFIX/bin/svhip
echo "SHARE_DIR=$CONDA_PREFIX/share/svhip" >> $CONDA_PREFIX/bin/svhip
head -1 $CONDA_PREFIX/share/svhip/run_svhip.sh >> $CONDA_PREFIX/bin/svhip
chmod +x $CONDA_PREFIX/bin/svhip
