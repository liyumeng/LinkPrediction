

if [ ! -d "utils/LINE" ]; then
    cd utils
    git clone https://github.com/tangjianpku/LINE.git
    cd ..
fi

cp utils/train_LINE.sh utils/LINE/linux/train_LINE.sh

if [ ! -f "data/tmp/author_network.txt" ]; then
    python process_network_data.py
fi

cd utils/LINE/linux
./train_LINE.sh ../../../data/tmp/node_network.txt ../../../data/features/node_network.bin
cd ../../../

cd utils/LINE/linux
./train_LINE.sh ../../../data/tmp/author_network.txt ../../../data/features/author_network.bin
cd ../../../

python test.py

