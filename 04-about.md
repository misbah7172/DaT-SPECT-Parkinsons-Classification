# DaT Parkinson's Challenge — About

Source: https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/page/988/

## About the data

### DaT scans

DaT scans are read as either normal or abnormal. Interpretation relies mainly on the visual assessment of morphological criteria such as asymmetry or intensity gradients in dopamine transporter uptake, sometimes supplemented by semi-quantitative measures like uptake ratios. When an exam is read as abnormal, it may confirm the neurodegenerative nature of a parkinsonian syndrome, providing valuable support for diagnosis and patient management.

Example image (normal DaT scan on left, pathological on right): https://s3.amazonaws.com/drivendata-public-assets/sfmn-scans-cropped.png

*A normal DaT scan (left) shows symmetric dopamine transporter uptake, while an abnormal scan (right) shows reduced and asymmetric uptake. Image courtesy of GaelO (https://gaelo.fr/en/).*

The roughly one in five exams that remain difficult to interpret with confidence, particularly atypical or early-stage cases, are where AI-assisted interpretation could add the most value and are the central motivation for this challenge.

### Creating the challenge dataset

The competition dataset consists of DaT scan examinations collected through a nationwide, multicenter collaboration across France. Each examination in the data set was reviewed and annotated by French experts.

Participating institutions include:

* Centre Hospitalier Universitaire de Toulouse
* Centre Hospitalier Régional Universitaire de Nancy
* Centre Hospitalier Régional Universitaire de Tours
* Centre Henri Becquerel
* Centre Hospitalier Régional et Universitaire de Brest
* Centre Hospitalier Universitaire Grenoble Alpes
* Hospices Civils de Lyon
* Centre Jean Perrin
* Assistance Publique – Hôpitaux de Marseille
* Institut Godinot

This collaborative effort has produced one of the largest multicenter collections of DaT imaging data assembled for machine learning research.

### Principal investigators

* Prof. Pierre Payoux (CHU de Toulouse)
* Prof. Antoine Verger (CHU de Nancy)
* Prof. Maria-Joao Santiago Ribeiro (CHU de Tours)
* Prof. Pierre Decazes (Centre Henri Becquerel)
* Dr. Solène Querellou (CHU de Brest)
* Dr. Nicolas de Leiris (CHU de Grenoble)
* Dr. Anthime Flaus (Hospices Civils de Lyon)
* Dr. Clément Bouvet (Centre Jean Perrin)
* Prof. Eric Guedj (AP-HM)
* Prof. Dimitri Papathanassiou (Institut Godinot)
* Dr. Caroline Bund (ICANS)

### Panel of experts

This data was reviewed and hand-labeled by the following experts:

* Dr. Solène Querellou
* Dr. Anthime Flaus
* Dr. Caroline Prunier
* Dr. Nicolas De Leiris
* Dr. Laura Rozenblum
* Prof. Maria Ribeiro
* Dr. Pierre Meneret
* Dr. Thibault Fidani
* Prof. Pierre Payoux
* Prof. Antoine Verger
* Dr. Tatiana Horowitz

## About the organizers

**French Society of Nuclear Medicine (SFMN)** (https://www.cnp-mn.fr/sfmn-accueil/): a professional society of French-speaking specialists dedicated to advancing nuclear medicine, molecular imaging, and related techniques. The society organized this challenge to accelerate progress on AI-assisted diagnosis of Parkinson's disease using DaT imaging — a core application area for nuclear medicine specialists.

The SFMN team for this challenge includes:

* Prof. Eric Guedj — Marseille University Hospitals (AP-HM); Head of the SFMN Neuro Working Group
* Prof. Florent Cachin — Centre Jean Perrin, Clermont-Ferrand; University Head of the Imaging Department
* Imad Bousaid — Project Manager

**GaelO** (https://gaelo.fr/en/): a unified platform for centralized imaging, DICOM orchestration, AI-powered analysis and clinical trial expertise, and assisted on data collection and harmonization for the challenge.

**Health Data Hub** (https://www.health-data-hub.fr/): a French federal agency responsible for fostering innovation and promoting artificial intelligence in the health sector, and supports data challenges that drive real-world impact in healthcare. The challenge is part of the Health Data Challenges call for projects supported by the France 2030 plan (https://anr.fr/en/france-2030/france-2030/).

## Learn more

The resources below provide a useful starting point for participants in understanding the clinical context, common imaging patterns, and prior approaches to automated classification.

* Booij, J., Habraken, J. B., Bergmans, P., Tissingh, G., Winogrodzka, A., Wolters, E. C., Janssen, A. G., Stoof, J. C., \& van Royen, E. A. (1998). Imaging of dopamine transporters with iodine-123-FP-CIT SPECT in healthy controls and patients with Parkinson's disease. *Journal of Nuclear Medicine, 39*(11), 1879–1884. PMID: 9829575 (https://pubmed.ncbi.nlm.nih.gov/9829575/)
* Ebrahimian Sadabad, F., Elahi, H., Bahrami, P., Mehrabi, S., \& Shahpasand, M. (2026). Detection of Parkinson's disease with neuroimaging modalities using machine learning and artificial intelligence: A systematic review. *Neurological Sciences*. https://doi.org/10.1007/s10072-025-08768-6
* Nazari, M., Eshghi, N., Zadeh, M. Z., Hajianfar, G., Sadeghi, R., \& Ahmadzadehfar, H. (2022). Explainable AI to improve acceptance of convolutional neural networks for automatic classification of dopamine transporter SPECT in the diagnosis of clinically uncertain parkinsonian syndromes. *European Journal of Nuclear Medicine and Molecular Imaging, 49*, 3521–3532. https://doi.org/10.1007/s00259-021-05569-9
* Quan, J., Xu, L., Xu, R., Tong, T., \& Su, J. (2019). *DaTscan SPECT Image Classification for Parkinson's Disease*. arXiv. https://arxiv.org/abs/1909.04142
* Zhao, Y., Wu, P., Wu, J., Brendel, M., Lu, J., Ge, J., Tang, C., Hong, J., Xu, Q., Liu, F., Sun, Y., Ju, Z., Lin, H., Guan, Y., Bassetti, C., Schwaiger, M., Huang, S.-C., Rominger, A., Wang, J., Zuo, C., \& Shi, K. (2022). Decoding the dopamine transporter imaging for the differential diagnosis of parkinsonism using deep learning. *European Journal of Nuclear Medicine and Molecular Imaging, 49*, 2798–2811. https://doi.org/10.1007/s00259-022-05804-x

