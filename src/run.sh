#!/bin/sh

# define path test:
FILE_IN="/hpc/zxio506/Atria_Data/Utah/CARMA1460/pre/lgemri.nrrd" # e.g. CARMA0440 CARMA0596 CARMA0606 CARMA0886 CARMA0937
#FILE_IN="/hpc/zxio506/Atria_Data/Waikato/00003/lgemri" # e.g. 00007 00007v2 000010 000010v2
#FILE_IN="/hpc/zxio506/Atria_Data/Kobe_University/15" # e.g. 2 5 9 11 15
#FILE_IN="/hpc/zxio506/Atria_Data/OSU_RussiaMRIs/OSU_RussiaMRIs_cavity_zh/joe/ori" # e.g. FA_lessgood Fed1 H_4779 joe
FILE_OUT="output_here"

# run prediction
python predict_ROI.py --path=$FILE_IN --out_dir=$FILE_OUT
python predict_Seg.py --path=$FILE_IN --out_dir=$FILE_OUT
