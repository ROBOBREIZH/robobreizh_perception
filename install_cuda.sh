mkdir -p install/ && cd install/
sudo apt-get install python-pip
pip install gdown

################################################
###      Install CUDA if not existing        ###
################################################
if type "nvidia-smi" > /dev/null; then
	echo -e "\n \e[43m Nvidia-smi installed, check Cuda version ... \e[0m \n"; 
	vss=$(nvidia-smi | grep "CUDA Version: 11.2")
	if [ "$vss" != "" ]; then 
	       echo -e "\n \e[42m Cuda 11.2 already installed, skipping \e[0m \n"; 
	else 
	       echo -e "\n \e[43m Installing Nvidia Cuda 11.2 Toolkit ... \e[0m \n"; 
	       wget https://developer.download.nvidia.com/compute/cuda/11.2.2/local_installers/cuda_11.2.2_460.32.03_linux.run
	       sudo sh cuda_11.2.2_460.32.03_linux.run
	       echo "export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}" >> ~/.bashrc
	       echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" >> ~/.bashrc
	       source ~/.bashrc
	fi
fi



