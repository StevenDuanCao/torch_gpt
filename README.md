# torch_gpt
A transformer language model architecture built from Pytorch. 

### Install
```bash
pip install torch pickle
```
### Training
Command to train and save model weights to `model_weights.pth`:
```bash
python train.py
```
### Example
Trained `GPT` class to generate Shakespeare-like text. Model is an  8-layer Transformer with 8 heads in each layer, a context window of 256 characters, and 1024 feature channels. Sample generation:
```
isabella:
so i perceived with his sone as moodlim
and titus years, sir, this as angelo will it,
whistsning i drop being in in friar'd;
in she all raise a noin to die:
'tis done, and let them fortune too;
do saint to their hatress spearf.
```
