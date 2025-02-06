Enhancements:
- set up 1 new Groq email for translation API key
- set up 2 new Groq email for second key rotation (use personal for sabrina)
- need to use Google colab, then load in the model manually.
  - Lik to model: https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt



Notes:
- MY and PH will need groq translation pipeline
- SG no need
- VN need huggingFace translation pipeline


Pipelines:
1. Pure English (SG)
2. Mixed English and Other (MY and PH)
3. Other (VN) (have to run this pipeline on colab notebook)


Overlap pipeline: 
- load_region_data() and format_llm_input() # for every region
- Here then split for diff translation methods