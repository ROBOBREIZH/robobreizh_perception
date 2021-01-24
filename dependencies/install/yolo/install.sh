#rm -rf darknet
cd ~/
git clone https://github.com/AlexeyAB/darknet.git
cp Makefile darknet/Makefile
cd darknet
make clean
make


