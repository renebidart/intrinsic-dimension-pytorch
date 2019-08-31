# first check if latent dim is very important, using decent sized intrinsic dim
python train_vae.py --bs 256 --epochs 60 --d 4096 --latent-size 8
python train_vae.py --bs 256 --epochs 60 --d 4096 --latent-size 12
python train_vae.py --bs 256 --epochs 60 --d 4096 --latent-size 16
python train_vae.py --bs 256 --epochs 60 --d 4096 --latent-size 32
python train_vae.py --bs 256 --epochs 60 --d 4096 --latent-size 64