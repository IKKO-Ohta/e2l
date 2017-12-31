#! /bin/csh -f

set train_dir="train"
set test_dir="test"
set tmp_dir="tmp-$$"

if( -d $train_dir || -d $test_dir || -d $tmp_dir ) then
    echo "*** Directory already exists."
    exit
endif

mkdir $train_dir
mkdir $test_dir

foreach file ( auto/djnml_daily_headline_splited/*.txt )
    set fname=`basename $file`
    ../program/format.pl < $file > $train_dir/$fname
end


mv $train_dir/2010-*.txt $test_dir
#foreach file ( $tmp_dir/2015-062[5-9].txt $tmp_dir/20150630.txt )
#    mv $file $test_dir
#end

