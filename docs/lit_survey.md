# Literature survey: "taste" in language and vision-language models (as of 2026-10-20)

How I checked: I confirmed arXiv IDs, titles and authors for nearly every item on the abstract page or the venue page. Items marked **(unverified)** I could not confirm. Items marked **(from memory)** are well-known papers whose IDs I did not re-fetch. Several 2026 items (arXiv IDs 26xx.xxxxx) are very recent, and I only read their abstracts.

---

## A. LLMs judging poetry and creative writing

**Human studies of AI poetry**
- **Porter & Machery 2024, *Sci. Rep.* 14:26133** ([PMC11564748](https://pmc.ncbi.nlm.nih.gov/articles/PMC11564748)).
  - 1,634 non-experts identified the author correctly 46.6% of the time, below chance. They judged AI poems (written in the style of 10 canonical poets) as human more often than the real poems.
  - A second group of 696 rated AI poems higher on rhythm, beauty and other qualities. The authors explain this by accessibility: AI poems are simpler to read.
- **Köbis & Mossink 2021, *Comput. Hum. Behav.* 114:106553** ([arXiv 2005.09980](https://arxiv.org/abs/2005.09980)). People could not reliably detect hand-selected GPT-2 poems, yet showed aversion to AI poetry. This partly contradicts Porter & Machery.
- **Hitsuwari, Ueda, Yun & Nomura 2023, *Comput. Hum. Behav.* 139:107502** (haiku). 385 raters each scored 40 haiku.
  - Haiku produced by AI with a human in the loop were rated highest. Haiku produced by AI alone were rated about the same as haiku by professional poets.
  - Raters could not tell the sources apart, and they gave lower scores to haiku they believed were AI-written.
- **Oddi, Scardapane, Sugimoto & Genovese 2026, "Authorship attribution and aesthetic evaluation of AI poetry: a case study with Haiku"** ([arXiv 2609.15511](https://arxiv.org/abs/2609.15511)).
  - Haiku from GPT-5, Gemini 2.5 and StableLM-7B were detected at chance (about 0.50). Haiku from Gemma-2B, LLaMA-2 and LLM-jp were somewhat detectable (0.59–0.67).
  - Ratings of fluency and "poeticness" predicted a "human" attribution but not whether the attribution was correct. This is closest to your haiku-by-model-size idea, but it uses detectability, not judge models.

**LLM judges compared with ground truth**
- **Chakrabarty et al. 2024, "Art or Artifice?" (CHI 2024)** ([arXiv 2309.14556](https://arxiv.org/abs/2309.14556)).
  - TTCW is a set of 14 binary Torrance-style tests. LLM stories passed 3–10 times fewer tests than stories by professionals.
  - LLM judges' correlation with the expert judgments was close to zero. Over 2,000 expert annotations are released.
- **Sawicki, Grześ, Brown & Góes 2025, EMNLP** ([arXiv 2502.19064](https://arxiv.org/abs/2502.19064)). This adapts the Consensual Assessment Technique (forced-choice ranking in small batches) to 90 poems whose ground truth is publication venue. Claude-3-Opus reached Spearman 0.87, against 0.38 for the best non-expert human.
- **LitBench (Fein, Russo, Xiang, Jolly, Rafailov & Haber; EACL 2026)** ([arXiv 2507.00769](https://arxiv.org/abs/2507.00769)). 2,480 test pairs of Reddit stories with debiased upvote labels.
  - Best zero-shot judge: Claude-3.7-Sonnet at 73%. Small open models: Gemma-3-12B 65.7%, Qwen-2.5-7B 59.9%, Llama-3.1-8B 58.1%.
  - Trained Bradley-Terry and generative reward models reach about 78%. Zero-shot accuracy "scales unpredictably" with model size, and chain-of-thought *hurt*.
- **Chakrabarty, Laban & Wu 2025, "AI-Slop to AI-Polish?"** ([arXiv 2504.07532](https://arxiv.org/abs/2504.07532)).
  - On the Writing Quality benchmark (4,729 judgments) frontier and reasoning LLMs "barely outperform random". Trained WQRM reward models reach 74%.
  - Nine expert writers preferred outputs selected by WQRM 66% of the time.
- **Chakrabarty, Ginsburg & Dhillon 2025** ([arXiv 2510.13939](https://arxiv.org/abs/2510.13939)). Experts preferred MFA writers over prompted frontier models. After fine-tuning on an author's complete works, experts preferred the AI text, mostly because clichéd "AI tics" disappeared.
- **Tutone, Franceschelli & Musolesi 2026** ([arXiv 2608.23705](https://arxiv.org/abs/2608.23705)). Across 11 creativity dimensions, LLM judges were misaligned with human ratings and systematically preferred AI stories. Objective metrics showed near-zero correlation with human ratings.
- **Tang-poetry study** ([arXiv 2510.15313](https://arxiv.org/abs/2510.15313)). LLM judges agreed with each other on "flawed standards" that diverged from two expert scholars.
- **POEMetric (Li, Wang & Wilkinson; ICLR 2026, OpenReview, no arXiv ID found).** 203 annotated human poems in seven fixed forms versus 6,090 poems from 30 LLMs.
  - Overall quality: humans 4.22, best LLM 3.20.
  - A snippet reports larger models within a family scoring higher (e.g. the Qwen2.5 0.5B–72B series), with Llama an exception.
- **Walsh, Preus & Antoniak 2024, "Sonnet or Not, Bot?" (Findings of EMNLP)** ([arXiv 2406.18906](https://arxiv.org/abs/2406.18906)). LLMs recognise fixed poetic forms well and unfixed forms poorly. A dataset of 4.1k poems with form labels is released.
- **Moëll 2026** ([arXiv 2607.20425](https://arxiv.org/abs/2607.20425)). DeepSeek ranked texts into six quality tiers at 79% accuracy. Its judgments were more sensitive to structure and voice than to vocabulary.
- **Klinge, Ortlieb & Koller 2026, "LLMs Generate Kitsch"** ([arXiv 2604.25929](https://arxiv.org/abs/2604.25929)). Readers rate LLM stories as more kitschy. The authors argue this explains why AI art is "rated higher yet feels hollow".

**Benchmarks**
- **EQ-Bench Creative Writing v3** (Sam Paech; [eqbench.com](https://eqbench.com/creative_writing.html)). Rubric scoring plus pairwise matchups rated with a Glicko-type system, judged by Claude, with explicit controls for length and "slop".
- **Judgemark v2.1** scores *judges* by separability, stability, and Kendall τ against the LMSys creative-writing Arena. It is a ready-made way to measure judge ability across model sizes.
- **WritingBench (NeurIPS 2025 D&B)** ([arXiv 2503.05244](https://arxiv.org/abs/2503.05244)). Rubrics generated per query plus a fine-tuned critic model.
- Not found: a paper titled "Can LLMs judge creative writing". The items above are the closest matches.

## B. VLM and MLLM aesthetic assessment

- **CLIP-based predictors**
  - **Hentschel, Kobs & Hotho 2022, "CLIP knows image aesthetics"**, *Front. AI* 5:976235. A linear probe on CLIP features beats ImageNet features on AVA, and zero-shot prompts already carry signal.
  - **LAION-Aesthetics predictor** ([blog](https://laion.ai/blog/laion-aesthetics/)). An MLP on CLIP ViT-L/14 image embeddings, trained on SAC, LAION-Logos and AVA.
  - **Taylor et al., FAccT 2026** ([arXiv 2601.09896](https://arxiv.org/abs/2601.09896)) audits that predictor. It favours images whose captions mention women and realistic Western and Japanese landscapes and portraits, and it filters out images whose captions mention men or LGBTQ+ people.
- **Nieto, Celona & Fernandez-Labrador 2022, "Understanding Aesthetics with Language" (NeurIPS D&B)** ([arXiv 2206.08614](https://arxiv.org/abs/2206.08614)). The RPCD dataset has 74K images and 220K Reddit critiques. The sentiment of a critique serves as a proxy for its aesthetic score.
- **Conwell et al. 2025, PNAS 122(4):e2306025121.** Vision models with no affect training linearly predict most of the explainable variance in human beauty, valence and arousal ratings.
- **Conwell et al. 2024** ([arXiv 2410.23603](https://arxiv.org/abs/2410.23603)). Unimodal vision features explain most of the variance in beauty ratings. Language alignment adds little.
- **Q-Future line (Wu et al.)**
  - **Q-Instruct** (CVPR 2024; [arXiv 2311.06783](https://arxiv.org/abs/2311.06783)).
  - **Q-Align / OneAlign** (ICML 2024; [arXiv 2312.17090](https://arxiv.org/abs/2312.17090)). Training on discrete rating words ("good", "poor") and reading out the expected score from token probabilities gives state of the art on IQA, IAA and VQA. This token-probability readout is a good protocol for your ratings.
  - **Q-Bench** (ICLR 2024; arXiv 2309.14181, from memory).
- **Aesthetic benchmarks and models**
  - **AesBench** ([arXiv 2401.08276](https://arxiv.org/abs/2401.08276)). 2,800 images annotated by experts. MLLMs show only "rudimentary" aesthetic perception.
  - **UNIAA** ([arXiv 2404.09619](https://arxiv.org/abs/2404.09619)).
  - **AesExpert** ([arXiv 2404.09624](https://arxiv.org/abs/2404.09624)).
  - **ArtiMuse / ArtiMuse-10K** (CVPR 2026; [arXiv 2507.14533](https://arxiv.org/abs/2507.14533)). Joint scoring plus an eight-attribute critique.
  - **AesTest** ([arXiv 2511.06360](https://arxiv.org/abs/2511.06360)). A secondary summary says Qwen2.5-VL at 32B gains little over 7B **(unverified in the paper)**.
  - **AesBiasBench (EMNLP 2025)** ([arXiv 2509.11620](https://arxiv.org/abs/2509.11620)). 19 MLLMs tested. Larger models align better with human preferences, and smaller ones show more demographic bias.
- **Jiang & Chen 2025, ACM MM** ([arXiv 2501.09012](https://arxiv.org/abs/2501.09012)). MLLMs hallucinate during aesthetic reasoning. Evidence-grounded "ArtCoT" prompting improves agreement with humans.
- **Abe, Daikoku & Kuniyoshi 2024** ([arXiv 2403.03594](https://arxiv.org/abs/2403.03594)). GPT-4V predicts both group-average and individual aesthetic ratings well.
- **Visual Aesthetic Benchmark (Feng et al. 2026)** ([arXiv 2605.12684](https://arxiv.org/abs/2605.12684)).
  - Experts agree much more on direct best/worst choices than on scalar scores.
  - The best of 20 frontier models picks both the best and the worst image correctly in only 26.5% of tasks, against 68.9% for experts.
  - Fine-tuning a 35B model on 2,000 expert examples brings it close to a 397B model.
- **Ghiasvand et al. 2026** ([arXiv 2606.29689](https://arxiv.org/abs/2606.29689)). MLLM photo critiques (7B–397B models, 1,227 posts from r/photocritique) are fluent and comprehensive rather than selective. LLM judges' preference rates for 7–8B critiques range from 9% to 81% depending on the judge.
- **Text-to-image preference and reward models**
  - **ImageReward** (Xu et al., NeurIPS 2023; [arXiv 2304.05977](https://arxiv.org/abs/2304.05977)).
  - **Pick-a-Pic / PickScore** ([arXiv 2305.01569](https://arxiv.org/abs/2305.01569)).
  - **HPSv2** ([arXiv 2306.09341](https://arxiv.org/abs/2306.09341)).
  - **HPSv3** (ICCV 2025; [arXiv 2508.03789](https://arxiv.org/abs/2508.03789)). Built on a VLM, with 1.17M pairwise comparisons.
  - **VQAScore** (ECCV 2024; [arXiv 2404.01291](https://arxiv.org/abs/2404.01291)). Measures prompt alignment, not aesthetics.
  - **MJ-Bench** (NeurIPS 2025 D&B; [arXiv 2407.04842](https://arxiv.org/abs/2407.04842)). Small CLIP-based scorers beat VLMs on image quality and alignment, while VLMs do better on safety and bias.
  - **Rich Human Feedback** (CVPR 2024; arXiv 2312.10240, from memory).
- **Marjieh et al. 2024, *Sci. Rep.*** ([arXiv 2302.01308](https://arxiv.org/abs/2302.01308)). LLM similarity judgments recover human perceptual structure (colour wheel, pitch). I found no paper titled "LLMs show human-like aesthetic …". This and AesBiasBench are the closest matches.

## C. Human vs AI art: discrimination and preference

- **ACX AI Art Turing Test.** The test was posted 2024-10-14 ([post](https://www.astralcodexten.com/p/ai-art-turing-test)); the results post is ["How Did You Do On The AI Art Turing Test?"](https://www.astralcodexten.com/p/how-did-you-do-on-the-ai-art-turing), Nov 2024.
  - About 11,000 participants classified 50 images, half human and half AI.
  - Median accuracy was 60%. Professional artists averaged 66%, and artists hostile to AI averaged 68%.
  - The most-liked picture was AI ("Riverside Cafe", Jack Galler), and 6 of the top 10 were AI.
  - Even the 1,278 respondents who rated AI art "1/5" had AI images among their blind favourites.
  - Raters judged by style: every Impressionist piece was called human except the one that actually was.
  - **Anonymised per-respondent data are public:** `slatestarcodex.com/Stuff/aituringpublic.xlsx`.
  - Caveats: the images were curated to remove obvious tells, and the sample is self-selected.
  - **I found no paper or post that ran VLMs on this set.** Searches of arXiv, LessWrong and Substack came up empty.
- **Label bias against AI art**
  - **Bellaiche et al. 2023, *Cogn. Res. P&I*.** The same art is rated lower on liking, beauty, profundity and worth when labelled "AI". The effect on sensory judgments is moderated by perceived story and effort.
  - **Millet et al. 2023, *Comput. Hum. Behav.* 143:107707.** In 4 experiments with 1,708 participants, an AI label lowered ratings via reduced perceived creativity and awe.
- **Khan et al. 2024, "AI Art Neural Constellation"** ([arXiv 2402.02453](https://arxiv.org/abs/2402.02453)). 6,000 WikiArt works and 3,200 AI works annotated for likability. Colour composition and familiar subjects drive liking, and AI art shows "reduced figuration".
- **Haverals & Martin 2025/26, "The human-authorship halo"** ([arXiv 2510.08831](https://arxiv.org/abs/2510.08831)). In Queneau-style passages, the pro-human label bias is +13.7 percentage points for humans and +34.3 points for 13 AI judges. A 14×14 matrix of AI evaluators by AI creators shows that judges devalue anything *labelled* AI, whichever model wrote it.
- **Fu et al. 2025** ([arXiv 2508.01408](https://arxiv.org/abs/2508.01408)). On about 40K paintings, VLMs are poor at attributing artists and at detecting AI-generated paintings.

## D. Interpretable features of aesthetic judgment and model internals

**Probing model internals (closest prior work to your step 2)**
- **Ryu & Yanaka 2026, "What Do VLMs Encode for Personalized Image Aesthetics Assessment?" (Findings of ACL 2026)** ([arXiv 2604.11374](https://arxiv.org/abs/2604.11374)). **This is the most direct precedent.**
  - Models: **Qwen3-VL 2B/4B/8B and Gemma 3 4B/12B**, with DINOv3 as a vision-only baseline.
  - Ridge probes on hidden states predict AADB attributes. Overall score reaches Spearman ≈0.70–0.73, versus 0.64–0.67 for DINOv3.
  - Text tokens in the middle of the language decoder carry the most information.
  - Linear models on hidden states beat the models' own verbal ratings for personalised scoring on PARA and LAPIS (ρ≈0.57–0.61, while text baselines have ρ≤0.26 on LAPIS).
- **Olson et al. 2024 (NeurIPS Creativity workshop)** ([arXiv 2412.06060](https://arxiv.org/abs/2412.06060)). A "creative minus boring" activation direction in Llama-3-8B correlates with human creativity judgments better than prompting the model, and steering along it increases creativity. This is the text-side precedent.
- **Cross-cultural VLM study, *Sci. Rep.* 2026** ([nature.com/articles/s41598-026-62173-3](https://www.nature.com/articles/s41598-026-62173-3)). Qwen3-VL-8B and Llama-3.2-11B-V score East Asian art lower than Western art. About 87% of the gap persists after controlling for spectral image statistics, and internal probes show a processing asymmetry. **(Partly verified: details come from search snippets only.)**
- **Interpretable assessment and reward models**
  - **Liu & Wagemans 2026** ([arXiv 2603.18108](https://arxiv.org/abs/2603.18108)). A subspace of aesthetic concepts plus a residual predictor.
  - **Soydaner & Wagemans 2024, *Br. J. Psychol.*** ([arXiv 2311.14410](https://arxiv.org/abs/2311.14410)). SHAP analysis of which attributes drive AADB, EVA and PARA scores.
  - **SARM** ([arXiv 2508.08746](https://arxiv.org/abs/2508.08746)) and **SparseRM** ([arXiv 2511.07896](https://arxiv.org/abs/2511.07896)). Reward models built on SAE features, so each reward can be attributed to features.
- **Tooling for decomposition and auto-labelling**
  - **SpLiCE** (NeurIPS 2024; [arXiv 2402.10376](https://arxiv.org/abs/2402.10376)). Decomposes CLIP embeddings into sparse combinations of concepts.
  - **Gandelsman, Efros & Steinhardt** (ICLR 2024; arXiv 2310.05916, from memory). Text-based decomposition of CLIP's image representation.
  - **Paulo et al.** (ICML 2025; [arXiv 2410.13928](https://arxiv.org/abs/2410.13928)). Open-source automatic labelling of SAE features.
  - Bills et al. 2023 (OpenAI) and Templeton et al. 2024 ("Scaling Monosemanticity"), both from memory.
  - **I found no paper applying PCA or SAEs with automatic labelling to *aesthetic/quality* axes, and none drawing a Pareto frontier.**
- **Bartho, Thoemmes & Redies 2023** ([arXiv 2307.00984](https://arxiv.org/abs/2307.00984)). Across 12 rated image databases, 20 statistical image properties and CNN features (reduced by PCA) predict ratings about equally well. Predictability differs a lot between photo and painting datasets.

**Human aesthetics and classic computational features**
- **Iigaya, Yi, Wahle, Tanwisuth & O'Doherty 2021, *Nat. Hum. Behav.* 5:743–755.** One shared set of interpretable low- and high-level features predicts art liking both within and across people (MTurk raters, about 1,500 according to press coverage). The predictive features emerge hierarchically in an object-recognition CNN.
- **Iigaya et al. 2023, *Nat. Commun.* 14:127.** fMRI shows the feature-to-value construction runs from visual cortex through parietal and lateral PFC to mPFC, resembling a deep CNN. Data: github.com/kiigaya/Art.
- **Jacobsen & Höfel 2002, *Percept. Mot. Skills* 95:755.** Symmetry, then complexity, were the strongest correlates of beauty. Some individuals used the cues in the opposite direction.
- **Van Geert & Wagemans 2020, *PACA* 14(2).** Review of order, complexity and their balance in aesthetic appreciation.
- **Brachmann & Redies 2017, *Front. Comput. Neurosci.* 11:102.** Review of computational aesthetics, including statistical image properties and fractal and complexity measures.
- **Parraga et al. 2025, "Aesthetics Without Semantics"** ([arXiv 2505.05331](https://arxiv.org/abs/2505.05331)). The MSC database has 10,426 images with minimal semantic content, each rated by 100 observers. **Restricting the range toward beautiful images can reverse feature–rating relationships**, so you need "ugly" items as well.

**Shared vs individual taste ("eye of the beholder")**
- **Vessel, Maurer, Denker & Starr 2018, *Cognition* 179:121.** Shared taste is high for faces and landscapes and low for architecture and artworks.
- **Hönekopp 2006, *JEP:HPP* 32:199.** For facial attractiveness, private and shared taste carry about equal weight.
- **Leder et al. 2016, *Front. Hum. Neurosci.* 10:155.** Private taste is about 75% for abstract art versus about 40% for faces.
- **Germine et al. 2015, *Curr. Biol.*** Face preferences are shaped mostly by environment, not genes. Two random people agree only about 48% of the time.
- **Specker et al. 2020, *PLoS ONE* 15:e0232083.** Agreement on aesthetic effects of artworks is low, lowest for liking, and expertise does not raise it.
- **Vessel et al. 2023, *Psych. Sci.* 34:1007.** Self-relevance predicts appeal. Shared taste explains only about 10–20% of reliable variance for art.
- **Poetry**
  - **Chaudhuri, Pickering, Dooley & Bhattacharya 2024, *PLoS ONE* e0307298.** 96 raters scored 36 poems; participant variance is large (ICC ≈0.3–0.4). Data are on OSF (osf.io/9mw7r).
  - **Chaudhuri et al., *PACA* (DOI 10.1037/aca0000649).** Aesthetic appeal is the strongest predictor of perceived poetic creativity.
- **Kao & Jurafsky 2012 (NAACL CLfL workshop) and 2015 (*LiLT* 12).** Award-winning poems differ from amateur ones mainly in **references to concrete objects and imagery**. This is a ready-made interpretable feature for poems.

## E. Self-preference and model-specific taste

- **Origins of the self-preference finding**
  - **Zheng et al. 2023** (MT-Bench; arXiv 2306.05685, from memory) first flagged "self-enhancement" bias.
  - **Panickssery, Bowman & Feng 2024 (NeurIPS oral)** ([arXiv 2404.13076](https://arxiv.org/abs/2404.13076)). GPT-4 recognises its own summaries 73.5% of the time. Self-recognition strength correlates linearly with self-preference after fine-tuning.
- **Mechanism and confounds**
  - **Wataoka, Takahashi & Ri 2024** ([arXiv 2410.21819](https://arxiv.org/abs/2410.21819)). Self-preference is driven by **perplexity/familiarity**: judges favour low-perplexity text whoever wrote it.
  - **Chen et al. 2025** ([arXiv 2504.03846](https://arxiv.org/abs/2504.03846)). Much of strong models' self-preference is *legitimate*, i.e. they really are better. The harmful part shows up where the judge errs, and long chain-of-thought mitigates it.
  - **Roytburg et al. 2026 (ICML)** ([arXiv 2601.22548](https://arxiv.org/abs/2601.22548)). With a judge-swap baseline, only 51% of previously reported self-preference cases stay significant.
  - **Davidson et al. 2024 (Findings of EMNLP)** ([arXiv 2407.06946](https://arxiv.org/abs/2407.06946)). No general self-recognition. Instead, models share a *consistent cross-model view of which models are best*.
- **Family and affinity bias**
  - **Spiliopoulou et al. 2025, "Play Favorites"** ([arXiv 2508.06709](https://arxiv.org/abs/2508.06709)). GPT-4o and Claude 3.5 favour themselves and their own family; Llama and Mistral do not.
  - **Li et al., ICLR 2026, "Preference Leakage"** ([arXiv 2502.01534](https://arxiv.org/abs/2502.01534)). Judges favour students distilled from them.
  - **Goel et al., ICML 2025** ([arXiv 2502.04313](https://arxiv.org/abs/2502.04313)). The CAPA similarity metric. Judges favour models *similar* to themselves (affinity bias), and frontier models' errors are converging.
  - **Awuni et al. 2026** ([arXiv 2609.17857](https://arxiv.org/abs/2609.17857)). A fully crossed design over Llama 3.1, Qwen 2.5, Gemma 2 and Yi 1.5. Same-family lift is +3.4 to +8.4 percentage points.
    - The naive per-family statistic correlates r = 0.95 with candidate ability, so you need their corrected estimator.
    - Adding the judge's likelihood advantage shrinks the effect by 61%.
- **Multimodal**
  - **Koyama et al. 2026** ([arXiv 2604.11589](https://arxiv.org/abs/2604.11589)). Across 1.29M caption–score pairs from 12 MLLMs there is self-preference and mutual preference within families (e.g. LLaVA). The Pomms ensemble reduces it.
  - **Xu et al. 2024, SIGIR** ([arXiv 2311.14084](https://arxiv.org/abs/2311.14084)). Retrievers rank AI-generated images above real ones. This is a "source bias" analogue for images.
- **AI–AI bias**
  - **Laurito et al. 2025, PNAS 122(31):e2415697122** ([arXiv 2407.12856](https://arxiv.org/abs/2407.12856)). LLMs prefer items described by LLMs (products, papers, films) over items described by humans.
  - **Jiang et al. 2025, "Artificial Hivemind" (NeurIPS 2025 D&B oral)** ([arXiv 2510.22954](https://arxiv.org/abs/2510.22954)). Strong homogeneity across models. LMs, reward models and judges are worst calibrated exactly where human annotators' preferences are idiosyncratic.
- **Not found: a VLM judging images from its own family's image generator, e.g. Qwen-VL rating Qwen-Image outputs.**

## F. Model size and scaling

- **Thakur et al. 2024/25 (GEM²)** ([arXiv 2406.12624](https://arxiv.org/abs/2406.12624)). Only the largest judges (70B, GPT-4) align well with humans. Small judges can still *rank* candidate models.
- **JudgeBench** ([arXiv 2410.12784](https://arxiv.org/abs/2410.12784)).
- **JudgeLM** ([arXiv 2310.17631](https://arxiv.org/abs/2310.17631)). Judges at 7B, 13B and 33B.
- **Prometheus** (ICLR 2024; arXiv 2310.08491, from memory) and **Prometheus 2** ([arXiv 2405.01535](https://arxiv.org/abs/2405.01535)).
- **Dorner, Nastl & Hardt, ICLR 2025** ([arXiv 2410.13341](https://arxiv.org/abs/2410.13341)). When the judge is no more accurate than the model it evaluates, debiasing can at most halve the ground-truth labels needed. This bears directly on small judges rating large generators.
- **Creative domains**
  - LitBench: zero-shot accuracy is non-monotonic in size, and small reward models (1–1.5B) can match large ones.
  - VAB: a 35B model fine-tuned on 2,000 expert examples approaches a 397B model.
  - AesBiasBench: bigger models are better aligned with human preferences.
  - Li et al. 2026 ([arXiv 2601.08003](https://arxiv.org/abs/2601.08003)): LLM-judge scores rise with model size, while novelty does not.
  - POEMetric: within-family size helps poem generation.
  - Marco et al. (COLING 2025; [arXiv 2409.11547](https://arxiv.org/abs/2409.11547)): a small fine-tuned BART beat humans on short synopses except on creativity.
  - Marco et al. "Pron vs Prompt" (EMNLP 2024; [arXiv 2407.01119](https://arxiv.org/abs/2407.01119)): GPT-4 is still far from a top novelist.
- **I found no "taste scaling law" paper and no joint sweep of generator size × judge size on the same items.**

## G. "Taste" as an AI capability (empirical)

- **Tong et al. 2026, "AI Can Learn Scientific Taste"** ([arXiv 2603.14473](https://arxiv.org/abs/2603.14473)). RL from community feedback, using citations as reward, trains a judge of research papers that generalises to future years.
- **Pan et al. 2026, "The Tasteful Agent"** ([arXiv 2609.25804](https://arxiv.org/abs/2609.25804)). Taste-Bench is built from decision forks in agent trajectories. The best model scores 59.7%, and a larger reasoning budget does not help.
- **Jaffe & Sherburn 2026, "TasteVal"** ([arXiv 2610.06824](https://arxiv.org/abs/2610.06824)). Experimental research taste is measured as a compute multiplier relative to 24 human experts. The top model reaches 2.3× the best expert. Since Dec 2025 the multiplier has doubled about every 3 months.
- See also Kitsch (A), Artificial Hivemind (E) and the human-authorship halo (C).

---

## Gaps / what would be novel in the planned experiment

1. **Generator size × judge size on the same items.** Nobody has crossed a size-sweep of generators (haiku from 0.5B to 400B models within one family) with a size-sweep of judges. Existing results are piecemeal and partly contradictory: LitBench is non-monotonic, POEMetric finds within-family gains, AesBiasBench finds bigger is better. Treating "model size as a quality proxy" also gives the wide quality range that Parraga et al. say is needed to avoid restriction-of-range artefacts. Validate the proxy with a small human rating set or WQRM-style reward models.
2. **Unsupervised axes plus automatic labels plus a Pareto frontier.** Ryu & Yanaka (2026) already show that Qwen3-VL and Gemma 3 hidden states linearly encode AADB attributes, and that probes beat the models' verbal ratings. You must cite and differentiate from them.
   - What is new: unsupervised discovery (PCA or SAE), automatic labelling (SpLiCE/Delphi-style), testing whether one "quality axis" is *shared between text and image*, and the Pareto analysis.
   - Also new: probe-vs-verbal comparisons for poems. Olson et al. is the only text precedent, and it covers creativity, not quality.
3. **ACX as a model benchmark.** No published model study uses it. The per-respondent xlsx lets you compare *model–human* agreement to the human–human noise ceiling, and model–model agreement to human–human agreement (Vessel/Hönekopp-style variance decomposition of shared versus private taste, for models).
   - Caveats: n = 50 and the images were curated. Risk of contamination: the images and answer key are public online, so check memorisation.
4. **Cross-model taste matrices in creative or aesthetic domains.** Self- and family-preference has been studied for summaries, question answering and captions. In creative work there is only Haverals & Martin's 14×14 matrix, which is label-driven.
   - A *blind* judge × generator matrix for haiku and images, with Awuni's family-fixed estimator, Roytburg's judge-swap baseline, and controls for the judge's log-probability (Wataoka), would be new.
   - VLM self- or family-preference for *generated images* appears unstudied.
5. **Confounds to design out:**
   - perplexity and familiarity, and length
   - "kitsch" and RLHF style
   - attribution labels: keep everything blind
   - position bias: Awuni reports 55% of pairs reverse when order is swapped
   - legitimate quality differences: Chen et al.; and the judge's own accuracy: Dorner et al.

## Recommended datasets & baselines

**Poems**
- POEMetric's 203 human poems in fixed forms, plus its LLM poems.
- The Sawicki et al. 90-poem set (labelled by venue tier).
- Kao & Jurafsky's award-winning vs amateur poems.
- Porter & Machery's stimuli and ratings (OSF availability **unverified**).
- Chaudhuri et al. 2024 OSF per-rater data, useful for shared vs private taste.
- Hitsuwari haiku (availability **unverified**).
- HF `p1atdev/modern_haiku` (Japanese, with editorial metadata) and `davanstrien/haiku_dpo`. The latter's scores are automatic form checks only.
- PoKi (61k children's poems by grade, a weak proxy for skill).
- TTCW, LitBench and the WQ benchmark for prose.
- Lamb et al. 2015 ICCC poem set **(unverified)**.

**Images**
- ACX xlsx.
- AVA, AADB (attributes), PARA (personal ratings plus attributes), EVA, TAD66K.
- Art: BAID, APDDv2 (NeurIPS 2024; [arXiv 2411.08545](https://arxiv.org/abs/2411.08545)), LAPIS (art with personal ratings; [arXiv 2504.07670](https://arxiv.org/abs/2504.07670)), ArtiMuse-10K, the Iigaya art data, VAB.
- MSC, for the "ugly" end of the range.
- Pick-a-Pic, HPDv3 and ImageReward data, which record which generator made each image and so suit eye-of-the-beholder matrices.

**Baselines**
- Images: LAION aesthetic MLP on CLIP ViT-L/14; linear probes on DINOv3 and CLIP (Hentschel; Ryu & Yanaka); Q-Align/OneAlign (read the expected score from token probabilities); ArtiMuse; HPSv3, PickScore and ImageReward.
- Text: WQRM, the LitBench Bradley-Terry reward model, EQ-Bench CW v3 rubric, Prometheus 2.
- Judge protocol: Sawicki's forced-choice consensual-assessment batches.
- Computational features:
  - images: symmetry, complexity, fractal dimension, colourfulness, the Redies statistical image properties
  - poems: concreteness and imagery, lexical diversity, semantic surprise
