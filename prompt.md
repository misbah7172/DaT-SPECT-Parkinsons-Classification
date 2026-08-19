analyze all this information to find out the best possible approach to get the best result.

1. participants are welcome to incorporate external datasets for training; however, use of external data is subject to important exceptions and caveats (see External Data section on the Home page, and the full Rules).
2. Each `.nii.gz` file contains a single 3D reconstructed volume stored as 16-bit unsigned integer voxel intensities.
3. images are **not** all the same size or resolution
4. **Volume dimensions** differ from scan to scan (for example, `142 × 142 × 112` or `128 × 128 × 128`). Your pipeline should not assume a fixed input shape.
5. The spacing is recorded in each file's NIfTI header and may be useful for resampling images to a common resolution.
6. image example : https://drivendata-public-assets.s3.amazonaws.com/dat\_example.png
7. Since the container will not have network access, all packages must be pre-installed. The runtime is built using uv (https://docs.astral.sh/uv/) and PyTorch (https://pytorch.org/) backed by CUDA 12.9. Package additions will be considered as long as they do not conflict and can build successfully. Python packages should be available via PyPI. To request an additional package be added to the runtime environment, follow the instructions in the runtime repository.
8. Object Segmentation
Determines which pixels belong to the same object.
Example: In medical imaging, adjacency rules decide if scattered cancer cells form one tumor or multiple.
9. Noise Removal
Isolates noisy pixels (e.g., salt-and-pepper noise) by checking if they’re disconnected from meaningful regions.


## Learn more

The resources below provide a useful starting point for participants in understanding the clinical context, common imaging patterns, and prior approaches to automated classification.

* Booij, J., Habraken, J. B., Bergmans, P., Tissingh, G., Winogrodzka, A., Wolters, E. C., Janssen, A. G., Stoof, J. C., \& van Royen, E. A. (1998). Imaging of dopamine transporters with iodine-123-FP-CIT SPECT in healthy controls and patients with Parkinson's disease. *Journal of Nuclear Medicine, 39*(11), 1879–1884. PMID: 9829575 (https://pubmed.ncbi.nlm.nih.gov/9829575/)
* Ebrahimian Sadabad, F., Elahi, H., Bahrami, P., Mehrabi, S., \& Shahpasand, M. (2026). Detection of Parkinson's disease with neuroimaging modalities using machine learning and artificial intelligence: A systematic review. *Neurological Sciences*. https://doi.org/10.1007/s10072-025-08768-6
* Nazari, M., Eshghi, N., Zadeh, M. Z., Hajianfar, G., Sadeghi, R., \& Ahmadzadehfar, H. (2022). Explainable AI to improve acceptance of convolutional neural networks for automatic classification of dopamine transporter SPECT in the diagnosis of clinically uncertain parkinsonian syndromes. *European Journal of Nuclear Medicine and Molecular Imaging, 49*, 3521–3532. https://doi.org/10.1007/s00259-021-05569-9
* Quan, J., Xu, L., Xu, R., Tong, T., \& Su, J. (2019). *DaTscan SPECT Image Classification for Parkinson's Disease*. arXiv. https://arxiv.org/abs/1909.04142
* Zhao, Y., Wu, P., Wu, J., Brendel, M., Lu, J., Ge, J., Tang, C., Hong, J., Xu, Q., Liu, F., Sun, Y., Ju, Z., Lin, H., Guan, Y., Bassetti, C., Schwaiger, M., Huang, S.-C., Rominger, A., Wang, J., Zuo, C., \& Shi, K. (2022). Decoding the dopamine transporter imaging for the differential diagnosis of parkinsonism using deep learning. *European Journal of Nuclear Medicine and Molecular Imaging, 49*, 2798–2811. https://doi.org/10.1007/s00259-022-05804-x


Winning solutions must be documented using the version of the Winning Model Documentation Template (https://github.com/drivendataorg/prize-winner-template/blob/main/example\_documentation\_guide.pdf) that will be provided to top-ranking participants to be eligible for recognition and prize money if offered.

MIT License (https://opensource.org/licenses/MIT)

