Train를 실행하기 위해서는 Jupyter notebook 환경을 추천합니다.

Jupyter notebook 환경이 아닐 시,

<명령어>
python train_256.py --lr 1e-4 --cv_dir checkpoints --batch_size 1 --data_dir ./data/ --alpha 0.8 --img_size 128 --test_epoch 10
python train_256_A2C.py --lr 1e-4 --cv_dir checkpoints --batch_size 1 --data_dir ./data/ --alpha 0.8 --img_size 128 --test_epoch 10

train_256.ipynb, trian_256_A2C.ipynb의 코드를 셀 단위로 순서대로 실행시키면 Set-up 부터 진행하도록 하였습니다.


Python 3.7 / CUDA 10.1 버전에서 모든 실행을 확인하였습니다.