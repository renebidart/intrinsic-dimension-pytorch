# check if epochs is very important, using decent sized intrinsic dim and latent size
python train_vae.py --bs 256 --epochs 15 --d 4096 --latent-size 16
python train_vae.py --bs 256 --epochs 30 --d 4096 --latent-size 16
python train_vae.py --bs 256 --epochs 60 --d 4096 --latent-size 16
python train_vae.py --bs 256 --epochs 200 --d 4096 --latent-size 16