# Online Handwriting recognition using Transformer
This is a part TensorFlow 2.x implementation of the transformer model mentioned in our paper 'Transformer-based Models for Arabic Online Handwriting Recognition'.
We trained the model in this repository on English Online Handwriting dataset [IAM](https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database) as a proof of concept. 

Paper link [here](https://thesai.org/Downloads/Volume13No5/Paper_102-Transformer_based_Models_for_Arabic_Online_Handwriting.pdf).



Model checkpoint [here](https://github.com/fakhralwajih/Online-Handwriting-recognition-using-Transformer/blob/main/Models/iam80.h5).


## Sample Predictions
| target                                         | prediction                                      | wer          | cer  |
| ---------------------------------------------- | ----------------------------------------------- | ------------ | ---- |
| except by roasting and boiling. When           | except by roasting and boiling. When            | 0            | 0.00 |
| Achilles gave a Royal feast the principal dish | Achilles gave a Royal feast the prinepal dishn. | 0.25         | 0.09 |
| the marks on the back                          | the marks on the back                           | 0            | 0.00 |
| foods for which they have                      | toods for which they have                       | 0.2          | 0.04 |
| Become the norm at Covent Garden.              | Become the norm al Count Garen.                 | 0.5          | 0.12 |
| home from school she had never grasped the     | home from school she had never grased he        | 0.25         | 0.05 |
| woman.                                         | worman.                                         | 1            | 0.17 |
| He beckoned over a waiter.                     | He beckoned over a waiter.                      | 0            | 0.04 |
| had been served he said: " I                   | had been served he said. " I                    | 0.1428571429 | 0.07 |
| not possess the lovely looks                   | not posess the bovely looks                     | 0.4          | 0.07 |
| reward. That means the                         | reward. That weaus the                          | 0.25         | 0.09 |
| ways of interpreting that loyalty              | ways of intopreting that loyalty                | 0.2          | 0.06 |
| These birds are, the                           | Theo birds are, the                             | 0.25         | 0.10 |
| Rob, what do you look for                      | Role, what do you look for                      | 0.1666666667 | 0.08 |
| comes, "he said lightly.                       | comes, "he said lightly.                        | 0            | 0.00 |
| her shroud of mist.                            | her shroud of mist.                             | 0            | 0.00 |
| Very interested, I was.                        | Very interestar. I was.                         | 0.25         | 0.17 |
| "What have you got in that                     | "What hare you got in that                      | 0.1666666667 | 0.04 |

## Usage

- Extract the zip files in the data directory. 
- Go through 0_data_preparation.ipynb notebook.
- Go through 1_model_tarining_testing.ipynb notebook.



## Citation
To cite this paper, please use:

```
@article{alwajih2022transformer,
  title={Transformer-based Models for Arabic Online Handwriting Recognition},
  author={Alwajih, Fakhraddin and Badr, Eman and Abdou, Sherif},
  journal={International Journal of Advanced Computer Science and Applications},
  volume={13},
  number={5},
  year={2022},
  publisher={Science and Information (SAI) Organization Limited}
}

```