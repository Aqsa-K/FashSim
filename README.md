# FashSim
An experimental project to find similarity between clothes based on their patterns and stitching style

Pakistani clothes are rich in patterns but they can still be put into distinct categories. This is an experimetnal project, speicifically for finding similarities in patetrns of pakistani clothes. 

The concept is inspired by Google SMILY project done for medical images; where patches are created and stored and KNN is used to fetch most similar patches given an input patch. 

This repo looks at something similar i.e patches for images are stored and KNN is used to fetch top 'n' patches given an input patch. 

Because we define different patterns as different styles, and style can be measured using Gram Matrix, this additional concept combined with cosine similarity is also used to find similar patches given an input patch. 

* VGG19 is used to get image embedding for the full clothes images
* A KNN is trained on these embedding then

     * given an input image, it is passed through VGG19 to get an image embedding which is then used as input to KNN to fetch 
                  the top 'n' similar image vectors. The corresponding images are treated as the similar images
                  
* Patches are created for all the images and a different KNN trained on these
* One of initial layers of VGG is used for the computation of style cost between different patches


### Results and Observations:

Observation in the test results show that

* the algorithm picks up color combinations as a similarity between clothes 
* in many cases the alogrithm is also able to pick shapes/structures (e.g triangular patterns) as a similarity between clothes/patches
* in many cases the alogirthm picked up similar pattern of flowers as a similarity between clothes
* When KNN is used on full images (and not on patches), the alogirthm is able to incorporate stitching style into similarity


#### Sample Results:

Input: 

![picture alt](https://github.com/Aqsa-K/FashSim/blob/master/results/test1.png)

Top Similar:

![picture alt](https://github.com/Aqsa-K/FashSim/blob/master/results/test1_r1.png)

![picture_alt](https://github.com/Aqsa-K/FashSim/blob/master/results/test1_r2.png)


Input:

![picture_alt](https://github.com/Aqsa-K/FashSim/blob/master/results/test2.png)

Top Similar:

![picture_alt](https://github.com/Aqsa-K/FashSim/blob/master/results/test2_r1.png)

![picture_alt](https://github.com/Aqsa-K/FashSim/blob/master/results/test2_r2.png)

Input:

![picture_alt](https://github.com/Aqsa-K/FashSim/blob/master/results/test3.png)

Top Similar:

![picture_alt](https://github.com/Aqsa-K/FashSim/blob/master/results/test3_r1.png)

Input:

![picture_alt](https://github.com/Aqsa-K/FashSim/blob/master/results/test5.png)

Top Similar:

![picture_alt](https://github.com/Aqsa-K/FashSim/blob/master/results/test5_r1.png)

### Extension:
* Find a way to combine both style of patterns and stitching style into a single similarity
* Ways to improve pattern style similarity

