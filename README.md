# Bhashaverse Documentation

[📝Paper](https://arxiv.org/pdf/2412.04351) | [💻Github Repo](https://github.com/vmujadia/onemtbhashaverse/tree/main) 

# About the model
<table border="1" cellpadding="6" cellspacing="0">
<tbody>
    <tr><td>Assamese (Bengali Script)</td><td>as</td><td>asm_Beng</td></tr>
    <tr><td>Awadhi (Devanagari Script)</td><td>aw</td><td>awa_Deva</td></tr>
    <tr><td>Bengali</td><td>bn</td><td>ben_Beng</td></tr>
    <tr><td>Bhojpuri</td><td>bh</td><td>bho_Deva</td></tr>
    <tr><td>Braj</td><td>br</td><td>bra_Deva</td></tr>
    <tr><td>Bodo</td><td>bx</td><td>brx_Deva</td></tr>
    <tr><td>Dogri</td><td>doi</td><td>doi_Deva</td></tr>
    <tr><td>Konkani (Devanagari Script)</td><td>go</td><td>gom_Deva</td></tr>
    <tr><td>Gondi</td><td>gn</td><td>gon_Deva</td></tr>
    <tr><td>Gujarati</td><td>gu</td><td>guj_Gujr</td></tr>
    <tr><td>Hindi</td><td>hi</td><td>hin_Deva</td></tr>
    <tr><td>Hinglish</td><td>hg</td><td>hingh_Deva</td></tr>
    <tr><td>Ho (Warang Citi Script)</td><td>hc</td><td>hoc_Wara</td></tr>
    <tr><td>Kannada</td><td>kn</td><td>kan_Knda</td></tr>
    <tr><td>Kashmiri (Arabic Script)</td><td>ks</td><td>kas_Arab</td></tr>
    <tr><td>Kashmiri (Devanagari Script)</td><td>ka</td><td>kas_Deva</td></tr>
    <tr><td>Khasi (Latin Script)</td><td>kh</td><td>kha_Latn</td></tr>
    <tr><td>Mizo (Latin Script)</td><td>lu</td><td>lus_Latn</td></tr>
    <tr><td>Maithili</td><td>ma</td><td>mai_Deva</td></tr>
    <tr><td>Magahi</td><td>mg</td><td>mag_Deva</td></tr>
    <tr><td>Malayalam</td><td>ml</td><td>mal_Mlym</td></tr>
    <tr><td>Marathi</td><td>mr</td><td>mar_Deva</td></tr>
    <tr><td>Manipuri (Bengali Script)</td><td>mn</td><td>mni_Beng</td></tr>
    <tr><td>Nepali</td><td>np</td><td>npi_Deva</td></tr>
    <tr><td>Oriya</td><td>or</td><td>ory_Orya</td></tr>
    <tr><td>Punjabi (Gurmukhi Script)</td><td>pa</td><td>pan_Guru</td></tr>
    <tr><td>Sanskrit</td><td>sa</td><td>san_Deva</td></tr>
    <tr><td>Santali (Ol Chiki Script)</td><td>st</td><td>sat_Olck</td></tr>
    <tr><td>Sinhala</td><td>si</td><td>sin_Sinh</td></tr>
    <tr><td>Sindhi (Arabic Script)</td><td>sn</td><td>snd_Arab</td></tr>
    <tr><td>Tamil</td><td>ta</td><td>tam_Taml</td></tr>
    <tr><td>Tulu (Kannada Script)</td><td>tc</td><td>tcy_Knda</td></tr>
    <tr><td>Telugu</td><td>te</td><td>tel_Telu</td></tr>
    <tr><td>Urdu</td><td>ur</td><td>urd_Arab</td></tr>
    <tr><td>English</td><td>en</td><td>eng_Latn</td></tr>
    <tr><td>Kangri</td><td>xr</td><td>xnr_Deva</td></tr>
</tbody>
</table>

# Table of contents

# Installation
- **Python version<=3.10 is recommended**.
- The repo uses **fairseq** to finetune the Bhashaverse model.

### Use the install.sh bash file to setup everything automatically.
```bash
# Clone the repo
git clone https://github.com/vmujadia/onemtbhashaverse.git
cd onemtbhashaverse

# Install all the dependencies and requirements along with model and tokenizer at once. 
source install.sh
```
### Manual Setup (*If above setup is not working for you*)
```bash
# Setup virtual env (You can use conda here if want)
virtualenv -p python3.8 bvenv
# If above line throws error then try
python3 -m venv bvenv
source bvenv/bin/activate

# Install the dependencies
pip3 install -r requirements.txt

# Install Fairseq
git clone https://github.com/pytorch/fairseq
cd fairseq
python3 -m pip install .

cd ..

# Download Models
mkdir models
python3 downloadmodels.py
```

# Data

- HuggingFace Dataset Link : https://huggingface.co/datasets/ltrciiith/bhashik-parallel-corpora-generic
- This huggingface dataset repo is gated, hence you will need to share your details by clicking *"Agree and Access Repository"*.
- Generate you huggingface accesstoken for accessing gated repos by clicking [here](https://huggingface.co/settings/tokens).
- Use below code to download data from HuggingFace.
  ```python
  import huggingface_hub
  ```
# Data Pre-processing

- Before sendiong the data to the translation model, it is required to tokenize the data.
- For tokenizing the it, the data needs to be in a specific way as mentioned below:
    #### Step-1 : Add Language Mappings
    * You will be preparing total 6 files for training: `train.src`, `train.tgt`, `valid.src`, `valid.tgt`, `test.src`, `test.tgt`.
    * Append language mappings before each sentence. It is the same format as you would have seen in flores.
    * For this, You need to append `###src_lang-tgt_lang### ` like language mapping before every source sentence, meaning in all .src files only. (***Remember to keep a space between language mapping and sentence***)
    > Example : 
        Original source sentence : भाषावर्स में आपका स्वागत है।
        Language mapping sentence : ###hin_Deva-guj_Gujr### भाषावर्स में आपका स्वागत है।
    #### Step-2 : Encode the mapped sentence using tokenizer
    * Use the `onemtv3b_spm.model` that you downloaded in setup, and encode all the sentences.
    * Below is a sample snippet refering to which you can perform both Step-1,2 together:
    ```python
    import sentencepiece as spm
    # Load the spm model
    sp = spm.SentencePieceProcessor('onemtv3b_spm.model')

    # Prepare a list of parallel sentences by appending language mappings and encoding them.
    train_lines = []
    for h,g in zip(hindi_sents, gujarati_sents):
        train_lines.append((' '.join(sp.encode('###hin_Deva-to-guj_Gujr### ' + h, out_type=str)), ' '.join(sp.encode(g, out_type=str))))

    # Save the files
    with open('train.src', 'w') as f: f.writelines(s + '\n' for s, t in train_lines)
    with open('train.tgt', 'w') as f: f.writelines(t + '\n' for s, t in train_lines)
    ```

    #### Step-3 : Preprocess the data using `fairseq-preprocess`

    * You need to use the fairseq-preprocess script to convert the resulting tokenized files into a binary data format. 
    * You should use the `dict.SRC.txt` and `dict.TGT.txt` files to ensure correct conversion of subwords to ids. 
    * A sample command is like so, modify as per your needs:
    ```bash
    fairseq-preprocess --source-lang src --target-lang tgt --trainpref train --validpref valid --testpref test --destdir data-bin --srcdict dict.SRC.txt --tgtdict dict.TGT.txt
    ```

# Training

* You can use fairseq-train script to train this model. 
* ***Make sure to supply the `custom_fairseq` directory as the `--user-dir` argument here in addition to whatever else your experiment requires.***
* You might want to move the model checkpoint to `checkpoints/checkpoint_last.pt` folder so that the training script automatically continues from there when run from the folder containing the checkpoints folder. 
* Or you can provide the path to the checkpoint with the appropriate flag. 
* A sample command is provided below so, modify as per your needs.
```bash
fairseq-train data-bin --user-dir custom_fairseq --arch transformer_18_18 --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr 5e-5 --lr-scheduler inverse_sqrt --warmup-updates 4000 --dropout 0.3 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4096 --skip-invalid-size-inputs-valid-test --max-epoch 101 --reset-optimizer
```

# Inference

* After you have fine-tuned/trained the model using all the steps mentioned above you need to specify the fine-tuned model's path in the config file.
* Open the `oneconfig.py` file where you can see below lines:
```python
TRANSLATION_MODEL_FOLDER="models/"
TRANSLATION_MODEL_PATH="models/onemtv3b.pt"

# CHANGE THESE TWO LINES TO:
TRANSLATION_MODEL_FOLDER="checkpoints/" # Or whatever folder name you are using to save your models
TRANSLATION_MODEL_PATH="checkpoints/checkpoint_last.pt"
```

* After this, use the `call_onemt.py` file to infer your fine-tuned model.
* Use the `translate_onemt()` function in this file to infer.
