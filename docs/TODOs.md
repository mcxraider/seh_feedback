Enhancements:
- set up 1 new Groq email for translation API key
- set up 2 new Groq email for second key rotation (use personal for sabrina)
- use open source huggingface model for translation


Notes:
- MY and PH will need groq translation pipeline
- SG no need
- VN need huggingFace translation pipeline


Pipelines:
1. Pure English (SG)
2. Mixed English and Other (MY and PH)
3. Other (VN)

Overlap pipeline: 
- load_region_data() and format_llm_input() # for every region
- Here then split for diff translation methods



