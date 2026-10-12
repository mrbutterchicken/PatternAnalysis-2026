# Feasibility Check-off - Option 4: Image Generation (Synthetic Medical Cohorts & Privacy)

### User Need, Scope & Acceptance Criteria
For open research, clinical MRI scans are required and the availability thereof is limited by patient privacy requirements and other things. This will be addressed with the training and utilisation of a StyleGAN-2 neural network to generate clinically feasible images.

The ADNI team was contacted for access and ADNI Guest access was granted. Data from 500 patients was donwloaded.  

The scope of this project has been limited in the following ways:
+ A single model is intended to be trained on Sagittal axis MPRAGE brain MRI scans with T1 weighting **only**. Further models can be trained separately (if time permits) to learn generation of other types of scans.
+ Although a full MRI scan covers every slice of the head, scans with very little tissue (e.g. close to the ears) are excised from training data. Additionally, the model does not learn to attach a slice position to its generated images. Thus, it is only capable of performing independent single-slice generation, not a consistent 200-slice MRI for a single synthetic person. 
+ The model is not trained on raw DICOM files, rather processed PNG files. The model also outputs PNG files only.
+ Given access to A100 GPUs on Google Colab Pro, we will ensure model training is stopped before 100 compute units are utilised (approximately 18 hours of training). 

### Model Choice & Course Concepts

The dataset is a list of 16,068 processed Sagittal-axis MPRAGE (T1 weighting) brain MRIs pre-processed into PNG format. The pre-processing code is included but should only be run once. 

The choice of model is a StyleGAN-2 as suggested in the problem specification (`Hard` difficulty band). Generative Adversarial Networks have been covered extensively by the course content. GANs are a very standard and effective method of image generation. 

To ensure non-memorisation of training data (i.e. model collapse) we send a mini-batch of generated images to the Discriminator along with the variance of the mini-batch, so the Discriminator can immediately detect collapse by near-zero variance. This immediately and heavily penalises the Generator for memorising training data. 

### Preliminary Feasibility Evidence

We ran 15 epochs (out of a proposed 300) of training and took random samples from the Generator's latent space at 5-epoch intervals. 

**Epoch 5:**

<img src="StyleGAN-s4801123/data/samples/epoch0005.png" alt="Project Diagram" width="350">

<br>

**Epoch 10:**

<img src="StyleGAN-s4801123/data/samples/epoch0010.png" alt="Project Diagram" width="350">

<br>

**Epoch 15:**

<img src="StyleGAN-s4801123/data/samples/epoch0015.png" alt="Project Diagram" width="350">

As we can see, there is significant progress in only 15 epochs, with anatomical structure being almost totally respected, and we are beginning to reproduce a lot of detail in the folds of the brain (although there is more work to do there). We also see significant standard deviation so model collapse does not seem to be a problem yet. 

### Risks, Budget & Fallback

1. 
    + *Risk:* model training goes over compute budget
    + *Mitigation:* ...buy more compute units lowkey
2. 
    + *Risk:* GAN sucks. 
    + *Mitigation:* ... find out why it sucks and report on it??

If GAN convergence is truly difficult, we can revert to a VAE, which worked super well in Demo 2 so should work well again. On that note, the GAN behaved really nicely in Demo 2 as well so lowkey I don't see why it should misbehave here. It already seems to behave pretty nicely. 
