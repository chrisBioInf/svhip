#!/bin/bash

THIS_DIR=$(basename $(pwd))
cd ..
mkdir -p $CONDA_PREFIX/share
cp -r $THIS_DIR $CONDA_PREFIX/share/Svhip
mkdir -p $CONDA_PREFIX/bin/
echo "#!/bin/sh" > $CONDA_PREFIX/bin/svhip
echo "SHARE_DIR=$CONDA_PREFIX/share/Svhip" >> $CONDA_PREFIX/bin/svhip
head -1 $CONDA_PREFIX/share/Svhip/run_svhip.sh >> $CONDA_PREFIX/bin/svhip
chmod +x $CONDA_PREFIX/bin/svhip
