# Awesome-Image Captioning
A paper list of image captioning as supplementary reference to this short survey. Based on this survey, we combed the papers and its codes in the field of IC in recent years.

This paper list is organized as follows:

Ⅰ. the existing surveys in IC field

Ⅱ. three main directions of current IC:

Nowadays, mainstream of IC model is heterogenous encoder-decoder architecture with three major improvement directions.

​		visual feature: advancement of encoder(CNN)

​		attention mechanism: changes in the attended source;  modification of  the architecture of the attention module

​		visual and language structure: explorations of structural inductive bias

Ⅲ. Transformer & homogenous architecture 

Many remarkable improvements in performance have achieved after the advent of Transformer.  
Thanks to the  architectural advantages  of Transformer, a promising pure Transformer-based homogeneous encoder-decoder captioner is around the corner.

Ⅳ. large scale pretraining

Motivated by NLP , researchers  in  the  vision-language  domain  also  proposed  to train  the  large-scale  Transformer  architectures.
Some of these multi-modal large-scale pre-training models can also be used for IC and have achieved much better performances than small-scale ones.

## Survey

- **A comprehensive survey of deep learning for image captioning.** MD  Zakir  Hossain. | [[pdf]](https://arxiv.org/pdf/1810.04020.pdf)

- **From Show to Tell: A Survey on Image Captioning.** Matteo Stefanini. | [[pdf]](https://arxiv.org/pdf/2107.06912.pdf)

- **Transformers in vision:  A survey.** Salman Khan. | [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3505244)

- **Pre-train,  prompt,  and  predict:   A  systematic  survey  of prompting methods in natural language processing.** Pengfei Liu.|[[pdf]](https://arxiv.org/pdf/2107.13586.pdf)



## Current Image Captioning



### Classic Encoder-Decoder Captioner

- **Sequence to sequence learning with neural networks.** Ilya  Sutskever. | [NIPS'14]| [[pdf]](https://proceedings.neurips.cc/paper/2014/file/a14ac55a4f27472c5d894ec1c3c743d2-Paper.pdf)
- **Cider: Consensus-based  image  description  evaluation.** Ramakrishna Vedantam. |evaluation metrics, CIDEr| [[pdf]](https://openaccess.thecvf.com/content_cvpr_2015/papers/Vedantam_CIDEr_Consensus-Based_Image_2015_CVPR_paper.pdf)

### Visual Feature -- CNN

- **Show and tell: A neural image caption generator.** Oriol    Vinyals. |image-level; GoogLeNet & pretrained by classification on ImageNet|[CVPR'15]| [[pdf]](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf)
- **Show, attend and tell: Neural  image  caption  generation  with  visual  attention.** Kelvin   Xu. | grid feature; classification; hard attention & soft attention | [[pdf]](http://proceedings.mlr.press/v37/xuc15.pdf)
- **Bottom-up and top-down attention for image captioning and visual question answering.**  Peter Anderson. |regional feature; object detection| [[pdf]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Anderson_Bottom-Up_and_Top-Down_CVPR_2018_paper.pdf)
- **Neural baby talk.**  Jiasen Lu. |attribute  classification|[[pdf]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Lu_Neural_Baby_Talk_CVPR_2018_paper.pdf)
- **Deconfounded image captioning: A causal retrospect.** Xu Yang. | DIC; a rethink on dataset bias| [[pdf]](https://arxiv.org/pdf/2003.03923.pdf)
- **Women also snowboard:  Overcoming bias in captioning models.**  Lisa Anne Hendricks. | solution to dataset bias | [[pdf]](https://openaccess.thecvf.com/content_ECCV_2018/papers/Lisa_Anne_Hendricks_Women_also_Snowboard_ECCV_2018_paper.pdf)
- **In defense of grid features for visual question answering.** Huaizu  Jiang. | [[pdf]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jiang_In_Defense_of_Grid_Features_for_Visual_Question_Answering_CVPR_2020_paper.pdf)
- 

### Attention Mechanism

- **Neural  machine  translation by jointly learning to align and translate.** Dzmitry Bahdanau. | for the first time introduced attention into NLP field | [[pdf]](https://arxiv.org/pdf/1409.0473.pdf)
- **Image captioning with semantic attention.**  Quanzeng You. |directly attend to semantic tags| [[pdf]](https://openaccess.thecvf.com/content_cvpr_2016/papers/You_Image_Captioning_With_CVPR_2016_paper.pdf)
- **Boosting image captioning with attributes.** Ting Yao. | [[pdf]](https://openaccess.thecvf.com/content_ICCV_2017/papers/Yao_Boosting_Image_Captioning_ICCV_2017_paper.pdf)
- **Sca-cnn:  Spatial and channel-wise attention in convolutional networks  for  image  captioning.**  Long Chen. | features from multi-channels| [[pdf]](https://openaccess.thecvf.com/content_cvpr_2017/papers/Chen_SCA-CNN_Spatial_and_CVPR_2017_paper.pdf)
- **Recurrent fusion network for image captioning.**  Wenhao Jiang. |multi-CNNs| [[pdf]](https://openaccess.thecvf.com/content_ECCV_2018/papers/Wenhao_Jiang_Recurrent_Fusion_Network_ECCV_2018_paper.pdf)
- **Reflective decoding network for image  captioning.** Lei Ke. |[[pdf]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Ke_Reflective_Decoding_Network_for_Image_Captioning_ICCV_2019_paper.pdf)
- **Look back and predict forward in image captioning.** Yu Qin. | [[pdf]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Qin_Look_Back_and_Predict_Forward_in_Image_Captioning_CVPR_2019_paper.pdf)
- **Meshed-memory transformer for image captioning.**  Marcella   Cornia. |use different sources as Q,K,V；augmented memory| [[pdf]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cornia_Meshed-Memory_Transformer_for_Image_Captioning_CVPR_2020_paper.pdf)
- **Attention on attention for image captioning.** Lun Huang. | AoA| [[pdf]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_Attention_on_Attention_for_Image_Captioning_ICCV_2019_paper.pdf)
- **X-linear attention networks for image captioning.** Yingwei Pan. |X-LAN|[[pdf]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Pan_X-Linear_Attention_Networks_for_Image_Captioning_CVPR_2020_paper.pdf)
- **Self-critical sequence training for image captioning.**  Steven  J  Rennie. |reinforcement learning-based strategy| [[pdf]](https://openaccess.thecvf.com/content_cvpr_2017/papers/Rennie_Self-Critical_Sequence_Training_CVPR_2017_paper.pdf)

### Visual and Language Structure -- Inductive Bias

- **Exploring  visual  relationship  for  image  captioning.**  Ting Yao.|scene graphs|[[pdf]](https://openaccess.thecvf.com/content_ECCV_2018/papers/Ting_Yao_Exploring_Visual_Relationship_ECCV_2018_paper.pdf)
- **Auto-encoding and distilling scene graphs for image captioning.** Xu  Yang. | [[pdf]](https://arxiv.org/pdf/1812.02378.pdf)
- **Relational inductive biases, deep learning, and graph networks.** Peter  W  Battaglia. |use GNN to embed relational inductive bias| [[pdf]](https://arxiv.org/pdf/1806.01261.pdf)
- **Say as you wish:  Fine-grained control of image caption generation with abstract scene graphs.**  Shizhe Chen. | [[pdf]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_Say_As_You_Wish_Fine-Grained_Control_of_Image_Caption_Generation_CVPR_2020_paper.pdf)
- **Hierarchy parsing for image captioning.**  Ting Yao. |tree-based encoder|[[pdf]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yao_Hierarchy_Parsing_for_Image_Captioning_ICCV_2019_paper.pdf)
- **Auto-parsing network for image captioning and visual question answering.**  Xu Yang. |text pattern |[[pdf]](https://openaccess.thecvf.com/content/ICCV2021/papers/Yang_Auto-Parsing_Network_for_Image_Captioning_and_Visual_Question_Answering_ICCV_2021_paper.pdf)
- **Knowing  when  to  look:  Adaptive attention via a visual sentinel for image captioning.** Jiasen  Lu. |design two modules for vision and non-vision words|[[pdf]](https://openaccess.thecvf.com/content_cvpr_2017/papers/Lu_Knowing_When_to_CVPR_2017_paper.pdf)
- **Learning to collocate neural modules for image captioning.** Xu Yang. |four modules : object, attribute, relation, and function|[[pdf]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yang_Learning_to_Collocate_Neural_Modules_for_Image_Captioning_ICCV_2019_paper.pdf)
- 

## Transformer & Homogenous Architecture

-  **Attention is all you need.** Ashish Vaswani. |[NIPS'17]|[[pdf]](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
-  **Swin transformer:  Hierarchical vision transformer using shifted windows.** Ze Liu. | visual encoder is a pre-trained vision Transformer| [[pdf]](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.pdf)
-  **Tree transformer: Integrating tree structures into self-attention.** Yau-Shian Wang. |[[pdf]](https://arxiv.org/pdf/1909.06639.pdf)

## Large Scale Pretraining

- **Bert:  Pre-training  of deep bidirectional transformers for language understanding.** Jacob  Devlin. |[[pdf]](https://arxiv.org/pdf/1810.04805.pdf)
- **Learning  transferable  visual  models  from  natural  language  supervision.** Alec Radford. |CLIP| [ICML'21]|[[pdf]](https://arxiv.org/pdf/2103.00020.pdf)
- **Vilbert: Pretraining  task-agnostic  visiolinguistic representations for vision-and-language tasks.** Jiasen Lu. |[NIPS'19]|[[pdf]](https://proceedings.neurips.cc/paper/2019/file/c74d97b01eae257e44aa9d5bade97baf-Paper.pdf)
- **E2e-vlp: End-to-end vision-language pre-training enhanced by visual learning.** Haiyang Xu. | [[pdf]](https://arxiv.org/pdf/2106.01804.pdf)
- **Language  models  are  unsupervised  multitask  learners.** Alec  Radford. |pre-train,  prompt,  and  predict|[[pdf]](http://www.persagen.com/files/misc/radford2019language.pdf)
- **Language  models  as knowledge  bases?** Fabio Petroni. | [[pdf]](https://arxiv.org/pdf/1909.01066.pdf)
- **Oscar:  Object-semantics aligned pre-training for vision-language tasks.** Xiujun Li. | [[pdf]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123750120.pdf)
- **Uniter: Learning universal image-text representations.** Yen-Chun Chen.| Masked  Region  Classification, Masked Region Feature Regression, and Masked Region Classification | [[pdf]](https://www.researchgate.net/profile/Linjie-Li-6/publication/336084110_UNITER_Learning_UNiversal_Image-TExt_Representations/links/5e14ea4da6fdcc2837619f52/UNITER-Learning-UNiversal-Image-TExt-Representations.pdf)
- **Align  before  fuse: Vision and  language  representation  learning  with  momentum distillation.** Junnan Li. | [[pdf]](https://proceedings.neurips.cc/paper/2021/file/505259756244493872b7709a8a01b536-Paper.pdf)
- **Unifying vision-and-language tasks via text generation.**  Jaemin Cho. |unifies various vision-language tasks.| [[pdf]](http://proceedings.mlr.press/v139/cho21a/cho21a.pdf)
- **Multimodal  few-shot  learning  with  frozen language models.** Maria Tsimpoukelli. | [[pdf]](https://proceedings.neurips.cc/paper/2021/file/01b7575c38dac42f3cfb7d500438b875-Paper.pdf)
-  **Simvlm: Simple visual language model pretraining with weak supervision.** Zirui Wang.| [[pdf]](https://arxiv.org/pdf/2108.10904.pdf)
- **Vqa: Visual question answering.**  Stanislaw Antol.|VQA|[[pdf]](https://openaccess.thecvf.com/content_iccv_2015/papers/Antol_VQA_Visual_Question_ICCV_2015_paper.pdf)
- **Open-vocabulary object detection using captions.**  Alireza Zareian.| [[pdf]](https://openaccess.thecvf.com/content/CVPR2021/papers/Zareian_Open-Vocabulary_Object_Detection_Using_Captions_CVPR_2021_paper.pdf)
- **Zero-shot text-to-image generation**. Aditya Ramesh. | [[pdf]](http://proceedings.mlr.press/v139/ramesh21a/ramesh21a.pdf)


