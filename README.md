# ViTCAP

 This repo contains the code for CVPR-2022 paper [Injecting Semantic Concepts into End-to-End Image Captioning](https://arxiv.org/abs/2112.05230).

 <img src="images/ViTCAP.png" width="650"> 

 ViTCAP is an end-to-end transformer-based image captioning model. ViTCAP takes the raw images as input and predict: 
 1. Semantic concepts exist in the image, and 
 2. an open-form textual description of the image. This repo contains the implementation and evaluation of ViTCAP on COCO-captioning dataset.
 
 
 ## ViTCAP Demo
  
  Please see [Loading Script.ipynb](Loading%20Script.ipynb) for a quick demo to load a trained ViTCAP checkpoint for inference.
  
 
 ## Dependencies
  The project enviroment in my local is PyTorch 1.6:
  
  `conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch`
  
  `pip install -r requirements.txt`

 ## Dataset
  
 Download the [COCO-captioning TSV files]() and place it in `./data/coco_caption`. 
 (**Please do not distribute it for commercial use due to possible copyright concern.** Otherwise I might have to retract it soon which causes great inconvenience for future researchers.)
   
 For the large scale pre-training corpus of ViTCAP, refer to [VinVL Repo](https://github.com/microsoft/Oscar/blob/master/VinVL_DOWNLOAD.md) for downloading large scale VL corpus.
 
 I do not plan to further update the VL pre-trained ViTCAP and their evaluations on other benchmarks at this time. Please refer to other implementations like [OSCAR/VinVL](https://github.com/microsoft/Oscar) for extension.

 ## Training & Evaluation
 The bellowing commands will do the training & evaluation.
 
 To conduct the CIDEr optimization, run the following command. Note that CIDEr optimization consumes large memories and I just randomly sample just 200 tokens for training on my V100 with 32GB memories. This probably indicates that better CIDEr scores might be reached using a larger memory device or better sampling method.
 
 
 ## Checkpoint
 
 Download the COCO-caption pre-trained checkpoint [here]() (Cross-entropy optimized).
 
 Download the concept classification trained ViT checkpoint. We find that it is essential to use the semantic classification optmized ViT to initialize ViTCAP, which is trained to predict the image-level concepts with 4 million images.
 
 The training log of can be found in [here](./checkpoint/Logit_Vilt_captioning_testing_batch-size_512_encoder_vit_base_patch16_384_lr_1e-4_iter_60_vitbfocal20_bert_tokenizer_tags_ENC-DEC_multiplier_0.1_expand_tag-classifier_emb.txt). The results across difference epochs are shown as below:
 
 <img src="images/map_TaxCocoCaption_test_Bleu_4.png" width="180">   <img src="images/map_TaxCocoCaption_test_CIDEr.png" width="180"> 
 <img src="images/map_TaxCocoCaption_test_METEOR.png" width="180">   <img src="images/map_TaxCocoCaption_test_ROUGE_L.png" width="180"> 
 <img src="images/map_TaxCocoCaption_test_SPICE.png" width="180"> 
 
 From left to right: <em> BLEU-4, CIDEr, METEOR, ROUGE, SPICE </em>.
    
 ## ToDo
- [x] Training and evaluation code
- [x] Quick demo notebook code.
- [x] COCO training TSV file.
- [x] COCO pre-trained checkpoint.
- [ ] Visualization codes for predicted semantic concepts and grounded visual concepts.
- [ ] Implementation on no-caps and Google-CC and pre-training. No plan at this time. 
 
 
 ## Citation
  
 Please cite our work if you find it helpful:
  
```bibtex
@inproceedings{fang2021injecting,
title={Injecting Semantic Concepts into End-to-End Image Captioning},
author={Zhiyuan Fang, Jianfeng Wang, Xiaowei Hu, Lin Liang, Zhe Gan, Lijuan Wang, Yezhou Yang, Zicheng Liu},
booktitle = {CVPR},
year = {2022},
}
```

## Acknowledgments
This implementation is largely based on [Jianfeng]()'s efforts and [Microsoft Azure-Florence Group](https://www.microsoft.com/en-us/research/project/project-florence-vl/). Thanks my collaborators.


## License
ViTCAP is released under the MIT license.


