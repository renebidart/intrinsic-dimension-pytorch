# check intrinsic dim using decent latent size and epochs
python train_vae.py --bs 256 --epochs 60 --d 128 --latent-size 16
python train_vae.py --bs 256 --epochs 60 --d 256 --latent-size 16
python train_vae.py --bs 256 --epochs 60 --d 512 --latent-size 16
python train_vae.py --bs 256 --epochs 60 --d 1024 --latent-size 16
python train_vae.py --bs 256 --epochs 60 --d 4096 --latent-size 16
python train_vae.py --bs 256 --epochs 60 --d 16384 --latent-size 16
