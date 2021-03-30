#rm -rf darknet

cd ~/
git clone https://github.com/AlexeyAB/darknet.git
cd -
if [ "$1" = "rtx" ]; then
	cp ./RTX/Makefile ~/darknet/Makefile
elif [ "$1" = "gtx" ]; then
	cp ./GTX_TITAN/Makefile ~/darknet/Makefile
else 
	echo "Error, please try again"
	exit 1
fi
cd ~/darknet
sudo make clean
sudo make


